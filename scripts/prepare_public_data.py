#!/usr/bin/env python3
"""Download and normalize public proxy datasets for the math-agent benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.adapters import (
    deduplicate,
    normalize_proofnet,
    normalize_scibench,
    normalize_supergpqa,
    normalize_theoremqa,
)
from benchmark.io import write_jsonl

THEOREMQA_URL = "https://raw.githubusercontent.com/TIGER-AI-Lab/TheoremQA/main/theoremqa_test.json"
PROOFNET_URL = "https://raw.githubusercontent.com/marcusm117/ProofNet-Verified/main/data/proofnet-verified.jsonl"
SCIBENCH_URL = "https://raw.githubusercontent.com/mandyyyyii/scibench/main/dataset/original/{name}.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare public math proxy datasets.")
    p.add_argument("--output-dir", default="data/public_candidates")
    p.add_argument("--sources", default="theoremqa,proofnet,scibench,supergpqa")
    p.add_argument("--supergpqa-dataset", default="m-a-p/SuperGPQA")
    p.add_argument("--limit-per-source", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="cap each source at 50 normalized rows")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def _get_json(url: str, timeout: int) -> Any:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_jsonl(url: str, timeout: int) -> Iterable[Mapping[str, Any]]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    for line_no, line in enumerate(response.text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{url}: invalid JSONL at line {line_no}") from exc


def _cap(rows: Iterable[dict | None], limit: int) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if row is None:
            continue
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def load_theoremqa(timeout: int, limit: int) -> list[dict]:
    return _cap((normalize_theoremqa(x) for x in _get_json(THEOREMQA_URL, timeout)), limit)


def load_proofnet(timeout: int, limit: int) -> list[dict]:
    return _cap((normalize_proofnet(x) for x in _get_jsonl(PROOFNET_URL, timeout)), limit)


def load_scibench(timeout: int, limit: int) -> list[dict]:
    rows: list[dict] = []
    for name in ("calculus", "diff"):
        for item in _get_json(SCIBENCH_URL.format(name=name), timeout):
            row = normalize_scibench(item, name)
            if row is not None:
                rows.append(row)
                if limit and len(rows) >= limit:
                    return rows
    return rows


def load_supergpqa(dataset_id: str, limit: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install data dependencies: pip install -r requirements-data.txt") from exc

    dataset = None
    last_error: Exception | None = None
    for split in ("train", "test"):
        try:
            dataset = load_dataset(dataset_id, split=split, streaming=True)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if dataset is None:
        raise RuntimeError(f"Unable to stream {dataset_id}: {last_error}")
    return _cap((normalize_supergpqa(x) for x in dataset), limit)


def summarize(rows: list[dict]) -> dict:
    return {
        "n_rows": len(rows),
        "by_domain": dict(sorted(Counter(r["domain"] for r in rows).items())),
        "by_answer_type": dict(sorted(Counter(r["answer_type"] for r in rows).items())),
        "by_review_status": dict(sorted(Counter(r["review_status"] for r in rows).items())),
    }


def main() -> None:
    args = parse_args()
    requested = {x.strip().lower() for x in args.sources.split(",") if x.strip()}
    valid = {"theoremqa", "proofnet", "scibench", "supergpqa"}
    unknown = requested - valid
    if unknown:
        raise SystemExit(f"Unknown sources: {', '.join(sorted(unknown))}")

    limit = 50 if args.smoke else args.limit_per_source
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, list[dict]] = {}
    if "theoremqa" in requested:
        loaded["theoremqa"] = load_theoremqa(args.timeout, limit)
    if "proofnet" in requested:
        loaded["proofnet_verified"] = load_proofnet(args.timeout, limit)
    if "scibench" in requested:
        loaded["scibench"] = load_scibench(args.timeout, limit)
    if "supergpqa" in requested:
        loaded["supergpqa"] = load_supergpqa(args.supergpqa_dataset, limit)

    merged: list[dict] = []
    report: dict[str, Any] = {"sources": {}}
    for source, rows in loaded.items():
        rows = deduplicate(rows)
        for idx, row in enumerate(rows):
            row["source_row_idx"] = idx
        write_jsonl(output_dir / f"{source}.jsonl", rows)
        report["sources"][source] = summarize(rows)
        merged.extend(rows)

    merged = deduplicate(merged)
    for idx, row in enumerate(merged):
        row["idx"] = idx
    write_jsonl(output_dir / "all_public_candidates.jsonl", merged)
    report["merged"] = summarize(merged)
    (output_dir / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote normalized public candidates to {output_dir}")


if __name__ == "__main__":
    main()
