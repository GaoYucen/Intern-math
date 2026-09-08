#!/usr/bin/env python3
"""Targeted R1 closure diagnostic on the four 8192-token hard cases.

This is intentionally isolated from the production agent. It preserves the
current transport-retry policy and varies only prompt closure pressure and
max_tokens so we can choose a production change causally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import ChatCompletionError, InternChatClient
from user_agent import R1_SYSTEM_PROMPT

HARD_CASE_IDX = [3, 10, 29, 40]

COMMIT_FIRST_PROMPT = R1_SYSTEM_PROMPT + """

Delivery-priority rules for long reasoning:
8. Your primary objective is to deliver the best justified answer before the inference limit, not to exhaust every possible derivation.
9. As soon as you have a defensible candidate answer, commit to it. Do not restart, re-derive, or perform optional verification that risks losing the final answer.
10. For objective-answer problems, prefer a short decisive derivation. If the reasoning is becoming long, skip nonessential exposition and checks.
11. Never consume the remaining budget on exploration after a candidate answer has been obtained. Immediately end with the required `FINAL_ANSWER:` line.
12. If uncertainty remains near the budget limit, submit the best justified candidate rather than continuing to search.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", choices=["current", "commit_first"], required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--concurrency", type=int, default=2)
    return p.parse_args()


def load_rows() -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    input_rows = [
        json.loads(x)
        for x in Path("data/benchmark_v1/input.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    gold_rows = [
        json.loads(x)
        for x in Path("data/benchmark_v1/gold.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    return ({int(r["idx"]): r for r in input_rows}, {int(r["idx"]): r for r in gold_rows})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def solve_one(
    client: InternChatClient,
    semaphore: asyncio.Semaphore,
    item: Dict[str, Any],
    prompt: str,
    max_tokens: int,
    output_dir: Path,
) -> None:
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.chat,
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": item["problem"]},
                ],
                0.0,
                max_tokens,
                thinking_mode=True,
            )
            if not isinstance(response, str) or not response.strip():
                raise ValueError("non-text or empty completion")
            telemetry = client.get_last_response_meta()
            record = {
                "idx": item["idx"],
                "status": "success",
                "final_response": response.strip(),
                "trace": [
                    {
                        "step": "r1_closure_diagnostic",
                        "content": {
                            "semantic_request_count": 1,
                            "prompt_variant": "diagnostic",
                            "max_tokens": max_tokens,
                            "client_telemetry": telemetry,
                        },
                    }
                ],
            }
        except ChatCompletionError as exc:
            record = {
                "idx": item["idx"],
                "status": "error",
                "final_response": "",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "trace": [
                    {
                        "step": "r1_closure_diagnostic_error",
                        "content": {
                            "semantic_request_count": 1,
                            "max_tokens": max_tokens,
                            "client_telemetry": exc.telemetry,
                        },
                    }
                ],
            }
        except Exception as exc:
            record = {
                "idx": item["idx"],
                "status": "error",
                "final_response": "",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "trace": [],
            }
        await asyncio.to_thread(write_json, output_dir / f"{item['idx']}.json", record)
        print(f"finished idx={item['idx']} status={record['status']}")


async def main_async(args: argparse.Namespace) -> None:
    inputs, gold = load_rows()
    selected_inputs = [inputs[i] for i in HARD_CASE_IDX]
    selected_gold = [gold[i] for i in HARD_CASE_IDX]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle_dir = out.parent / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "input.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected_inputs),
        encoding="utf-8",
    )
    (bundle_dir / "gold.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected_gold),
        encoding="utf-8",
    )

    system_prompt = R1_SYSTEM_PROMPT if args.prompt == "current" else COMMIT_FIRST_PROMPT
    client = InternChatClient()
    semaphore = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(
        *[
            solve_one(client, semaphore, item, system_prompt, args.max_tokens, out)
            for item in selected_inputs
        ]
    )

    rows: List[Dict[str, Any]] = []
    for idx in HARD_CASE_IDX:
        rows.append(json.loads((out / f"{idx}.json").read_text(encoding="utf-8")))

    summary: Dict[str, Any] = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "n": len(rows),
        "success": sum(r.get("status") == "success" for r in rows),
        "errors": sum(r.get("status") != "success" for r in rows),
        "final_marker_count": 0,
        "length_finish_count": 0,
        "http_attempts": 0,
        "retries": 0,
        "items": [],
    }
    for r in rows:
        text = str(r.get("final_response", ""))
        meta: Dict[str, Any] = {}
        for t in r.get("trace", []):
            maybe = t.get("content", {}).get("client_telemetry")
            if isinstance(maybe, dict):
                meta = maybe
        marker = "FINAL_ANSWER:" in text.upper()
        finish = meta.get("finish_reason")
        summary["final_marker_count"] += int(marker)
        summary["length_finish_count"] += int(finish == "length")
        summary["http_attempts"] += int(meta.get("http_attempt_count", 0) or 0)
        summary["retries"] += int(meta.get("retry_count", 0) or 0)
        summary["items"].append(
            {
                "idx": r.get("idx"),
                "status": r.get("status"),
                "final_marker": marker,
                "finish_reason": finish,
                "response_chars": len(text),
                "usage": meta.get("usage"),
                "http_attempt_count": meta.get("http_attempt_count"),
                "retry_count": meta.get("retry_count"),
                "total_latency_seconds": meta.get("total_latency_seconds"),
            }
        )
    summary["final_marker_rate"] = summary["final_marker_count"] / len(rows)
    write_json(out.parent / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
