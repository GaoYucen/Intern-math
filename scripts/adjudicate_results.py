#!/usr/bin/env python3
"""Adjudicate strict evaluator failures/reviews with a reference-based math judge.

The deterministic evaluator remains the first line of scoring. Only items that
are not already marked correct are sent to the judge (except strict wrong
multiple-choice answers, which remain wrong because label comparison is exact).
The judge may also flag a benchmark item as invalid when the problem/reference
pair is internally inconsistent or unusable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmark import read_jsonl, score_answer
from llm_client import InternChatClient


JUDGE_SYSTEM_PROMPT = """You are a strict independent grader for a mathematics benchmark.
Judge whether the candidate answer correctly solves the stated problem.

Rules:
1. Mathematical correctness is the only criterion; wording need not match the reference.
2. For numeric/symbolic questions, accept mathematically equivalent forms and normal rounding.
3. For proof questions, require the essential argument to be valid and sufficiently complete. A bare assertion is not enough unless the problem itself only asks for a short conclusion.
4. The reference answer is strong evidence, but some proof references are formal theorem statements rather than prose proofs. Use the problem statement itself as the primary specification.
5. Use verdict "invalid" only when the benchmark problem/reference pair itself is clearly broken, contradictory, missing essential information, or has an evidently wrong gold answer.
6. Use verdict "uncertain" only when you genuinely cannot determine correctness.
7. Do not give credit for a final answer that is mathematically wrong merely because intermediate reasoning is plausible.

Return exactly one JSON object and no markdown:
{"verdict":"correct|wrong|invalid|uncertain","reason":"brief reason"}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--report_dir", required=True)
    p.add_argument("--judge-model", default="intern-s1-pro")
    p.add_argument("--fallback-model", default="intern-s2-preview")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-candidate-chars", type=int, default=18000)
    return p.parse_args()


def make_client(model: str) -> InternChatClient:
    previous = os.environ.get("INTERN_MODEL")
    os.environ["INTERN_MODEL"] = model
    try:
        return InternChatClient(timeout=180, retry=3)
    finally:
        if previous is None:
            os.environ.pop("INTERN_MODEL", None)
        else:
            os.environ["INTERN_MODEL"] = previous


def parse_judge_json(text: str) -> tuple[str, str]:
    if not isinstance(text, str):
        return "uncertain", "judge returned non-text response"
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        return "uncertain", "judge did not return JSON"
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return "uncertain", "judge JSON parse failed"
    verdict = str(obj.get("verdict", "uncertain")).strip().lower()
    if verdict not in {"correct", "wrong", "invalid", "uncertain"}:
        verdict = "uncertain"
    reason = str(obj.get("reason", "")).strip()[:600]
    return verdict, reason


def compact_candidate(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    # Preserve both the beginning (setup) and the end (conclusion/final answer).
    half = max_chars // 2
    return text[:half] + "\n...[middle truncated by grader]...\n" + text[-half:]


def judge_item(
    row: dict[str, Any],
    result: dict[str, Any],
    primary: InternChatClient,
    fallback: InternChatClient | None,
    max_chars: int,
) -> dict[str, Any]:
    candidate = compact_candidate(result.get("final_response", ""), max_chars)
    prompt = (
        f"PROBLEM:\n{row.get('problem', '')}\n\n"
        f"ANSWER TYPE:\n{row.get('answer_type', 'text')}\n\n"
        f"REFERENCE ANSWER:\n{row.get('answer', '')}\n\n"
        f"CANDIDATE RESPONSE:\n{candidate}\n\n"
        "Grade the candidate according to the rubric."
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    used_model = primary.model
    try:
        response = primary.chat(messages, temperature=0.0, max_tokens=600, thinking_mode=False)
    except Exception as primary_exc:
        if fallback is None:
            return {
                "verdict": "uncertain",
                "reason": f"judge API failed: {primary_exc}",
                "judge_model": used_model,
            }
        used_model = fallback.model
        try:
            response = fallback.chat(messages, temperature=0.0, max_tokens=600, thinking_mode=False)
        except Exception as fallback_exc:
            return {
                "verdict": "uncertain",
                "reason": f"primary and fallback judge API failed: {fallback_exc}",
                "judge_model": used_model,
            }

    verdict, reason = parse_judge_json(response)
    return {"verdict": verdict, "reason": reason, "judge_model": used_model}


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")

    rows = read_jsonl(args.benchmark)
    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    primary = make_client(args.judge_model)
    fallback = None
    if args.fallback_model and args.fallback_model != args.judge_model:
        fallback = make_client(args.fallback_model)

    records: list[dict[str, Any]] = []
    tasks: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    strict_counts = Counter()

    for row in rows:
        path = out_dir / f"{row['idx']}.json"
        if not path.exists():
            result = {"status": "missing", "final_response": ""}
        else:
            result = json.loads(path.read_text(encoding="utf-8"))

        if result.get("status") != "success":
            strict_state = "error"
            strict_reason = result.get("error", {}).get("message", "")
            predicted = ""
        else:
            score = score_answer(
                result.get("final_response", ""),
                row.get("answer", ""),
                row.get("answer_type", "text"),
            )
            predicted = score.predicted
            strict_reason = score.reason
            strict_state = (
                "correct" if score.correct is True
                else "wrong" if score.correct is False
                else "review"
            )

        strict_counts[strict_state] += 1
        record = {
            "idx": row["idx"],
            "domain": row.get("domain", "unknown"),
            "source_dataset": row.get("source_dataset", "unknown"),
            "answer_type": row.get("answer_type", "text"),
            "strict_state": strict_state,
            "strict_reason": strict_reason,
            "predicted": predicted,
            "gold": row.get("answer", ""),
            "final_state": strict_state,
            "judge_model": "",
            "judge_reason": "",
        }
        records.append(record)

        # Exact multiple-choice label comparison is already decisive.
        exact_choice_wrong = strict_state == "wrong" and row.get("answer_type") in {
            "choice", "multiple_choice"
        }
        if result.get("status") == "success" and strict_state in {"wrong", "review"} and not exact_choice_wrong:
            tasks.append((row, result, record))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                judge_item, row, result, primary, fallback, args.max_candidate_chars
            ): record
            for row, result, record in tasks
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                judged = future.result()
            except Exception as exc:  # defensive; keep experiment moving
                judged = {
                    "verdict": "uncertain",
                    "reason": f"judge worker failed: {exc}",
                    "judge_model": "",
                }
            record["final_state"] = judged["verdict"]
            record["judge_model"] = judged.get("judge_model", "")
            record["judge_reason"] = judged.get("reason", "")

    final_counts = Counter(r["final_state"] for r in records)
    by_domain = defaultdict(Counter)
    by_answer_type = defaultdict(Counter)
    by_source = defaultdict(Counter)
    for r in records:
        by_domain[r["domain"]][r["final_state"]] += 1
        by_answer_type[r["answer_type"]][r["final_state"]] += 1
        by_source[r["source_dataset"]][r["final_state"]] += 1

    denominator = final_counts["correct"] + final_counts["wrong"]
    summary = {
        "n_items": len(records),
        "strict_counts": dict(strict_counts),
        "judge_primary_model": args.judge_model,
        "judge_fallback_model": args.fallback_model,
        "n_sent_to_judge": len(tasks),
        "final_counts": dict(final_counts),
        "effective_accuracy": final_counts["correct"] / denominator if denominator else None,
        "effective_denominator": denominator,
        "excluded_invalid": final_counts["invalid"],
        "unresolved_uncertain": final_counts["uncertain"],
        "errors": final_counts["error"],
        "by_domain": {k: dict(v) for k, v in sorted(by_domain.items())},
        "by_answer_type": {k: dict(v) for k, v in sorted(by_answer_type.items())},
        "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
        "method_note": (
            "Deterministic scoring is used first. Strict wrong/review items except exact "
            "multiple-choice mismatches are adjudicated by a reference-based math judge. "
            "Invalid/uncertain/error items are excluded from effective accuracy."
        ),
    }

    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (report_dir / "details.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with (report_dir / "adjudications.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            if r["judge_model"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
