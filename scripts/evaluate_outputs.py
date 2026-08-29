#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import read_jsonl, score_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--report_dir", default="reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.benchmark)
    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    details = []
    by_domain = defaultdict(Counter)
    total = Counter()

    for row in rows:
        output_path = out_dir / f"{row['idx']}.json"
        if not output_path.exists():
            result = {"status": "missing", "final_response": ""}
        else:
            result = json.loads(output_path.read_text(encoding="utf-8"))

        if result.get("status") != "success":
            state = "error"
            score = None
            pred = ""
            reason = result.get("error", {}).get("message", "")
        elif "answer" not in row:
            state = "unscored"
            score = None
            pred = result.get("final_response", "")
            reason = "benchmark row has no gold answer"
        else:
            s = score_answer(
                result.get("final_response", ""),
                row["answer"],
                row.get("answer_type", "text"),
            )
            pred = s.predicted
            reason = s.reason
            state = "correct" if s.correct is True else "wrong" if s.correct is False else "review"
            score = s.correct

        domain = row.get("domain", "unknown")
        total[state] += 1
        by_domain[domain][state] += 1
        details.append(
            {
                "idx": row["idx"],
                "domain": domain,
                "difficulty": row.get("difficulty", "unknown"),
                "answer_type": row.get("answer_type", "text"),
                "state": state,
                "predicted": pred,
                "gold": row.get("answer", ""),
                "reason": reason,
            }
        )

    auto_scored = total["correct"] + total["wrong"]
    summary = {
        "n_items": len(rows),
        "counts": dict(total),
        "auto_scored_accuracy": (total["correct"] / auto_scored) if auto_scored else None,
        "by_domain": {k: dict(v) for k, v in sorted(by_domain.items())},
        "note": "review items are intentionally excluded from automatic accuracy",
    }

    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (report_dir / "details.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=details[0].keys() if details else [])
        if details:
            writer.writeheader()
            writer.writerows(details)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
