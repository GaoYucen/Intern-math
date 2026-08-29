#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import has_explicit_final_answer, read_jsonl, score_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--report_dir", default="reports")
    return p.parse_args()


def _slice_summary(counter: Counter) -> dict:
    auto_scored = counter["correct"] + counter["wrong"]
    return {
        **dict(counter),
        "auto_scored": auto_scored,
        "auto_scored_accuracy": (counter["correct"] / auto_scored) if auto_scored else None,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.benchmark)
    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    details = []
    by_domain = defaultdict(Counter)
    by_answer_type = defaultdict(Counter)
    by_source = defaultdict(Counter)
    total = Counter()
    response_lengths = []
    success_lengths = []
    explicit_final_count = 0
    success_count = 0

    for row in rows:
        output_path = out_dir / f"{row['idx']}.json"
        if not output_path.exists():
            result = {"status": "missing", "final_response": ""}
        else:
            result = json.loads(output_path.read_text(encoding="utf-8"))

        final_response = result.get("final_response", "")
        if not isinstance(final_response, str):
            final_response = ""
        response_chars = len(final_response)
        response_lengths.append(response_chars)
        explicit_final = has_explicit_final_answer(final_response)
        explicit_final_count += int(explicit_final)

        if result.get("status") != "success":
            state = "error"
            pred = ""
            reason = result.get("error", {}).get("message", "")
        elif "answer" not in row:
            state = "unscored"
            pred = final_response
            reason = "benchmark row has no gold answer"
            success_count += 1
            success_lengths.append(response_chars)
        else:
            success_count += 1
            success_lengths.append(response_chars)
            s = score_answer(
                final_response,
                row["answer"],
                row.get("answer_type", "text"),
            )
            pred = s.predicted
            reason = s.reason
            state = "correct" if s.correct is True else "wrong" if s.correct is False else "review"

        domain = row.get("domain", "unknown")
        answer_type = row.get("answer_type", "text")
        source = row.get("source_dataset", "unknown")
        total[state] += 1
        by_domain[domain][state] += 1
        by_answer_type[answer_type][state] += 1
        by_source[source][state] += 1
        details.append(
            {
                "idx": row["idx"],
                "domain": domain,
                "source_dataset": source,
                "difficulty": row.get("difficulty", "unknown"),
                "answer_type": answer_type,
                "state": state,
                "explicit_final": explicit_final,
                "response_chars": response_chars,
                "predicted": pred,
                "gold": row.get("answer", ""),
                "reason": reason,
            }
        )

    auto_scored = total["correct"] + total["wrong"]
    n_items = len(rows)
    summary = {
        "n_items": n_items,
        "counts": dict(total),
        "success_rate": (success_count / n_items) if n_items else None,
        "explicit_final_rate": (explicit_final_count / success_count) if success_count else None,
        "auto_scored": auto_scored,
        "auto_scored_accuracy": (total["correct"] / auto_scored) if auto_scored else None,
        "response_chars": {
            "mean_all": statistics.mean(response_lengths) if response_lengths else None,
            "median_success": statistics.median(success_lengths) if success_lengths else None,
            "max_success": max(success_lengths) if success_lengths else None,
        },
        "by_domain": {k: _slice_summary(v) for k, v in sorted(by_domain.items())},
        "by_answer_type": {k: _slice_summary(v) for k, v in sorted(by_answer_type.items())},
        "by_source": {k: _slice_summary(v) for k, v in sorted(by_source.items())},
        "note": "proof/review items are excluded from automatic accuracy; API errors are reported separately",
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
