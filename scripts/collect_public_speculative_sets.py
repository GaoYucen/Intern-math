#!/usr/bin/env python3
"""Collect public challenge-associated math sets for *local* robustness testing.

The script deliberately does not ingest any asset that claims exact hidden-eval
question/answer correspondence. Third-party problem text is fetched at runtime
and written to a local/artifact directory rather than committed into this repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "fresh_holdout_v1" / "source_manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/fresh_holdout_v1/collected")
    p.add_argument("--keep-clones", action="store_true")
    return p.parse_args()


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _fingerprint(text: str) -> str:
    return hashlib.sha256(_norm_text(text).encode("utf-8")).hexdigest()[:20]


def _clone(repo: str, target: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", repo, str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        return False, f"clone exception: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, proc.stdout[-2000:]
    return True, proc.stdout[-1000:]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "problems", "items", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _normalize_row(row: dict[str, Any], source_name: str, source_file: str) -> dict[str, Any] | None:
    problem = row.get("problem") or row.get("question") or row.get("prompt")
    if not isinstance(problem, str) or not problem.strip():
        return None
    answer = row.get("answer")
    subject = row.get("subject") or row.get("domain") or row.get("category") or "unknown"
    external_id = row.get("idx", row.get("id", row.get("qid", "")))
    return {
        "fingerprint": _fingerprint(problem),
        "problem": problem.strip(),
        "answer": answer if isinstance(answer, (str, int, float, bool, list)) else None,
        "subject": str(subject),
        "source": source_name,
        "source_file": source_file,
        "external_id": external_id,
        "has_gold": answer is not None,
        "role": "external_public_speculative",
    }


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    work_root = Path(tempfile.mkdtemp(prefix="fresh-holdout-sources-"))
    merged: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"sources": [], "excluded_sources": manifest.get("excluded_sources", [])}

    try:
        for source in manifest.get("included_public_sources", []):
            name = source["name"]
            repo = source["repo"]
            clone_dir = work_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            ok, clone_log = _clone(repo, clone_dir)
            entry: dict[str, Any] = {
                "name": name,
                "repo": repo,
                "clone_ok": ok,
                "files": [],
                "clone_log_tail": clone_log,
            }
            if not ok:
                report["sources"].append(entry)
                continue

            for rel in source.get("files", []):
                path = clone_dir / rel
                file_report = {"path": rel, "exists": path.exists(), "rows": 0, "accepted": 0}
                if path.exists() and path.is_file():
                    try:
                        rows = _read_rows(path)
                        file_report["rows"] = len(rows)
                        for row in rows:
                            norm = _normalize_row(row, name, rel)
                            if norm is None:
                                continue
                            fp = norm["fingerprint"]
                            if fp not in merged:
                                merged[fp] = norm
                                file_report["accepted"] += 1
                    except Exception as exc:
                        file_report["error"] = f"{type(exc).__name__}: {exc}"
                entry["files"].append(file_report)
            report["sources"].append(entry)

        rows = list(merged.values())
        rows.sort(key=lambda x: (x["source"], x["source_file"], str(x["external_id"])))
        with (output_dir / "external_public.jsonl").open("w", encoding="utf-8") as f:
            for idx, row in enumerate(rows):
                row = {"idx": idx, **row}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        report["unique_problem_count"] = len(rows)
        report["with_gold"] = sum(bool(r.get("has_gold")) for r in rows)
        report["without_gold"] = len(rows) - report["with_gold"]
        (output_dir / "collection_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "unique_problem_count": report["unique_problem_count"],
            "with_gold": report["with_gold"],
            "without_gold": report["without_gold"],
            "output_dir": str(output_dir),
        }, ensure_ascii=False, indent=2))
    finally:
        if args.keep_clones:
            print(f"kept clones at {work_root}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
