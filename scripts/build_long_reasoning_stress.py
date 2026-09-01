#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("data/benchmark_v1_full.jsonl")
OUT = Path("data/long_reasoning_stress")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = read_jsonl(SRC)

    selected = []
    seen = set()

    def add(items):
        for row in items:
            key = row["idx"]
            if key not in seen:
                selected.append(row)
                seen.add(key)

    # 1) All peer-verified graduate PDE items. These were the clearest hard-case
    # failure signal in B0 (HardMath2 0/10).
    add(sorted((r for r in rows if r.get("source_dataset") == "hardmath2"), key=lambda r: r["idx"]))

    # 2) Formal/expert proof sources with stronger gold reliability than the
    # noisy theorem-style short-answer pool.
    add(sorted((r for r in rows if r.get("source_dataset") == "ma_proofbench"), key=lambda r: r["idx"])[:10])
    add(sorted((r for r in rows if r.get("source_dataset") == "proofnet_verified"), key=lambda r: r["idx"])[:10])

    # 3) High-failure DeepMath items, ranked by the source's failed_count signal.
    deep = [r for r in rows if r.get("source_dataset") == "deepmath_gap"]
    deep.sort(key=lambda r: (-int((r.get("source_meta") or {}).get("failed_count", 0)), r["idx"]))
    add(deep[:10])

    if len(selected) != 40:
        raise SystemExit(f"Expected exactly 40 stress items, got {len(selected)}")

    # Re-index locally so main.py output files are compact and deterministic,
    # while preserving the original benchmark index for paired analysis.
    input_rows = []
    gold_rows = []
    manifest_rows = []
    for local_idx, row in enumerate(selected):
        common = dict(row)
        common["original_idx"] = row["idx"]
        common["idx"] = local_idx

        input_rows.append({
            "idx": local_idx,
            "problem": row["problem"],
            "original_idx": row["idx"],
            "domain": row.get("domain"),
            "answer_type": row.get("answer_type"),
            "source_dataset": row.get("source_dataset"),
        })
        gold_rows.append(common)
        manifest_rows.append({
            "idx": local_idx,
            "original_idx": row["idx"],
            "source_dataset": row.get("source_dataset"),
            "domain": row.get("domain"),
            "answer_type": row.get("answer_type"),
            "difficulty": row.get("difficulty"),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "input.jsonl", input_rows)
    write_jsonl(OUT / "gold.jsonl", gold_rows)
    (OUT / "manifest.json").write_text(
        json.dumps({"n_rows": len(selected), "items": manifest_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    from collections import Counter
    print("n=", len(selected))
    print("source=", dict(Counter(r.get("source_dataset") for r in selected)))
    print("domain=", dict(Counter(r.get("domain") for r in selected)))
    print("answer_type=", dict(Counter(r.get("answer_type") for r in selected)))


if __name__ == "__main__":
    main()
