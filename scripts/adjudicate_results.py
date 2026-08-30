#!/usr/bin/env python3
"""Adjudicate strict evaluator failures/reviews with a reference-based math judge.

The deterministic evaluator remains the first line of scoring. Only items that
are not already marked correct are sent to the judge (except strict wrong
multiple-choice answers, which remain wrong because label comparison is exact).
The judge may also flag a benchmark item as invalid when the problem/reference
pair is internally inconsistent or unusable.

Judge output is intentionally parsed defensively. We prefer a simple VERDICT
line, but also accept JSON and a standalone verdict token. If a model produces a
verbose answer that cannot be parsed, one short normalization call is made before
falling back to ``uncertain``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import read_jsonl, score_answer
from llm_client import InternChatClient


VALID_VERDICTS = {"correct", "wrong", "invalid", "uncertain"}

JUDGE_SYSTEM_PROMPT = """You are a strict independent grader for a mathematics benchmark.
Judge whether the candidate answer correctly solves the stated problem.

Rules:
1. Mathematical correctness is the only criterion; wording need not match the reference.
2. For numeric/symbolic questions, accept mathematically equivalent forms and normal rounding.
3. For proof questions, require the essential argument to be valid and sufficiently complete. A bare assertion is not enough unless the problem itself only asks for a short conclusion.
4. The reference answer is strong evidence, but some proof references are formal theorem statements rather than prose proofs. Use the problem statement itself as the primary specification.
5. Use INVALID only when the benchmark problem/reference pair itself is clearly broken, contradictory, missing essential information, or has an evidently wrong gold answer.
6. Use UNCERTAIN only when you genuinely cannot determine correctness.
7. Do not give credit for a mathematically wrong final answer merely because intermediate reasoning is plausible.

Do not output analysis before the verdict. Return exactly these two lines:
VERDICT: CORRECT|WRONG|INVALID|UNCERTAIN
REASON: one brief reason
"""

REPAIR_SYSTEM_PROMPT = """Normalize a grader's already-written decision. Do not re-grade the mathematics.
Return exactly one line:
VERDICT: CORRECT|WRONG|INVALID|UNCERTAIN
If the prior grader output does not contain a recoverable decision, return VERDICT: UNCERTAIN.
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
    p.add_argument("--max-raw-excerpt", type=int, default=1800)
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


def _clean_verdict(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^a-z_]", "", token)
    aliases = {
        "incorrect": "wrong",
        "false": "wrong",
        "true": "correct",
        "valid": "correct",
    }
    token = aliases.get(token, token)
    return token if token in VALID_VERDICTS else None


def _json_dicts(text: str):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_judge_response(text: Any) -> tuple[str, str, bool, str]:
    """Return (verdict, reason, parsed, method)."""
    if not isinstance(text, str):
        return "uncertain", "judge returned non-text response", False, "non_text"

    raw = text.strip()
    if not raw:
        return "uncertain", "judge returned empty response", False, "empty"

    # 1) JSON anywhere in the response, including fenced markdown JSON.
    for obj in _json_dicts(raw):
        verdict = _clean_verdict(obj.get("verdict"))
        if verdict:
            reason = str(obj.get("reason", "")).strip()[:600]
            return verdict, reason, True, "json"

    # 2) Preferred explicit VERDICT line. Markdown decoration is tolerated.
    m = re.search(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*|__)?verdict(?:\*\*|__)?\s*[:：=\-]\s*"
        r"(?:\*\*|__|`)?\s*(correct|wrong|incorrect|invalid|uncertain|true|false)\b",
        raw,
    )
    if m:
        verdict = _clean_verdict(m.group(1)) or "uncertain"
        reason_match = re.search(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*|__)?reason(?:\*\*|__)?\s*[:：=\-]\s*(.+)$", raw)
        reason = reason_match.group(1).strip()[:600] if reason_match else ""
        return verdict, reason, True, "verdict_line"

    # 3) A standalone decision token/heading, e.g. **CORRECT**.
    for line in raw.splitlines():
        token_line = line.strip().strip("`*_#[](){}:：.- ")
        verdict = _clean_verdict(token_line)
        if verdict:
            return verdict, "", True, "standalone_token"

    return "uncertain", "judge verdict could not be parsed", False, "unparsed"


def compact_candidate(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[middle truncated by grader]...\n" + text[-half:]


def compact_excerpt(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    half = max_chars // 2
    return value[:half] + "\n...[excerpt truncated]...\n" + value[-half:]


def _call_judge(client: InternChatClient, messages: list[dict[str, str]], max_tokens: int) -> str:
    response = client.chat(messages, temperature=0.0, max_tokens=max_tokens, thinking_mode=False)
    return str(response)


def _repair_unparsed(client: InternChatClient, raw_response: str) -> tuple[str, str, bool, str, str]:
    repair_messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "PRIOR GRADER OUTPUT:\n" + compact_excerpt(raw_response, 6000),
        },
    ]
    repaired_raw = _call_judge(client, repair_messages, max_tokens=80)
    verdict, reason, parsed, method = parse_judge_response(repaired_raw)
    if parsed:
        return verdict, reason, True, "repair_" + method, repaired_raw
    return "uncertain", "judge verdict remained unparseable after normalization", False, "repair_failed", repaired_raw


def judge_item(
    row: dict[str, Any],
    result: dict[str, Any],
    primary: InternChatClient,
    fallback: InternChatClient | None,
    max_chars: int,
    max_raw_excerpt: int,
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

    attempts: list[tuple[str, InternChatClient]] = [(primary.model, primary)]
    if fallback is not None:
        attempts.append((fallback.model, fallback))

    last_error = ""
    last_raw = ""
    for used_model, client in attempts:
        try:
            raw = _call_judge(client, messages, max_tokens=350)
            last_raw = raw
        except Exception as exc:
            last_error = f"judge API failed for {used_model}: {exc}"
            continue

        verdict, reason, parsed, method = parse_judge_response(raw)
        if parsed:
            return {
                "verdict": verdict,
                "reason": reason,
                "judge_model": used_model,
                "parse_method": method,
                "raw_excerpt": compact_excerpt(raw, max_raw_excerpt),
            }

        # Most historical failures were formatting failures, not grading
        # uncertainty. Normalize the model's own conclusion with a very short
        # second call before trying another model.
        try:
            verdict, reason, parsed, method, repaired_raw = _repair_unparsed(client, raw)
        except Exception as exc:
            last_error = f"judge normalization failed for {used_model}: {exc}"
            continue
        if parsed:
            return {
                "verdict": verdict,
                "reason": reason,
                "judge_model": used_model,
                "parse_method": method,
                "raw_excerpt": compact_excerpt(raw, max_raw_excerpt),
                "repair_excerpt": compact_excerpt(repaired_raw, 500),
            }

    return {
        "verdict": "uncertain",
        "reason": last_error or "judge verdict could not be parsed",
        "judge_model": attempts[-1][0] if attempts else "",
        "parse_method": "failed",
        "raw_excerpt": compact_excerpt(last_raw, max_raw_excerpt),
    }


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
            "judge_parse_method": "",
            "judge_raw_excerpt": "",
            "judge_repair_excerpt": "",
        }
        records.append(record)

        exact_choice_wrong = strict_state == "wrong" and row.get("answer_type") in {
            "choice", "multiple_choice"
        }
        if result.get("status") == "success" and strict_state in {"wrong", "review"} and not exact_choice_wrong:
            tasks.append((row, result, record))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                judge_item,
                row,
                result,
                primary,
                fallback,
                args.max_candidate_chars,
                args.max_raw_excerpt,
            ): record
            for row, result, record in tasks
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                judged = future.result()
            except Exception as exc:
                judged = {
                    "verdict": "uncertain",
                    "reason": f"judge worker failed: {exc}",
                    "judge_model": "",
                    "parse_method": "worker_failed",
                    "raw_excerpt": "",
                }
            record["final_state"] = judged["verdict"]
            record["judge_model"] = judged.get("judge_model", "")
            record["judge_reason"] = judged.get("reason", "")
            record["judge_parse_method"] = judged.get("parse_method", "")
            record["judge_raw_excerpt"] = judged.get("raw_excerpt", "")
            record["judge_repair_excerpt"] = judged.get("repair_excerpt", "")

    final_counts = Counter(r["final_state"] for r in records)
    parse_methods = Counter(r["judge_parse_method"] for r in records if r["judge_model"])
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
        "judge_parse_methods": dict(parse_methods),
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
            "The judge uses a VERDICT-line protocol with JSON/standalone-token compatibility "
            "and one normalization retry for unparseable outputs. Invalid/uncertain/error "
            "items are excluded from effective accuracy."
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
