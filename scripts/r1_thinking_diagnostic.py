#!/usr/bin/env python3
"""Interleaved R1 thinking-mode diagnostic on four long-reasoning hard cases.

The production agent is not modified. This diagnostic keeps the current R1
system prompt and transport policy fixed while varying only thinking_mode and
max_tokens. Telemetry is captured inside the same worker thread that performs
the HTTP call so thread-local response metadata cannot be lost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import ChatCompletionError, InternChatClient
from user_agent import R1_SYSTEM_PROMPT

HARD_CASE_IDX = [3, 10, 29, 40]
VARIANTS: Dict[str, Dict[str, Any]] = {
    "thinking_true_8192": {"thinking_mode": True, "max_tokens": 8192},
    "thinking_false_8192": {"thinking_mode": False, "max_tokens": 8192},
    "thinking_false_6144": {"thinking_mode": False, "max_tokens": 6144},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="outputs/r1_thinking")
    p.add_argument("--summary-root", default="reports/r1_thinking/raw")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260908)
    return p.parse_args()


def load_rows() -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    inputs = [
        json.loads(x)
        for x in Path("data/benchmark_v1/input.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    gold = [
        json.loads(x)
        for x in Path("data/benchmark_v1/gold.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    return ({int(r["idx"]): r for r in inputs}, {int(r["idx"]): r for r in gold})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def solve_sync(
    client: InternChatClient,
    item: Dict[str, Any],
    variant_name: str,
    thinking_mode: bool,
    max_tokens: int,
) -> Dict[str, Any]:
    try:
        response = client.chat(
            [
                {"role": "system", "content": R1_SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
        )
        telemetry = client.get_last_response_meta()
        if not isinstance(response, str) or not response.strip():
            raise ValueError("non-text or empty completion")
        return {
            "idx": item["idx"],
            "status": "success",
            "final_response": response.strip(),
            "trace": [
                {
                    "step": "r1_thinking_diagnostic",
                    "content": {
                        "semantic_request_count": 1,
                        "variant": variant_name,
                        "thinking_mode": thinking_mode,
                        "max_tokens": max_tokens,
                        "client_telemetry": telemetry,
                    },
                }
            ],
        }
    except ChatCompletionError as exc:
        return {
            "idx": item["idx"],
            "status": "error",
            "final_response": "",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "trace": [
                {
                    "step": "r1_thinking_diagnostic_error",
                    "content": {
                        "semantic_request_count": 1,
                        "variant": variant_name,
                        "thinking_mode": thinking_mode,
                        "max_tokens": max_tokens,
                        "client_telemetry": exc.telemetry,
                    },
                }
            ],
        }
    except Exception as exc:
        return {
            "idx": item["idx"],
            "status": "error",
            "final_response": "",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "trace": [],
        }


async def run_one(
    client: InternChatClient,
    semaphore: asyncio.Semaphore,
    item: Dict[str, Any],
    variant_name: str,
    cfg: Dict[str, Any],
    output_root: Path,
) -> None:
    async with semaphore:
        record = await asyncio.to_thread(
            solve_sync,
            client,
            item,
            variant_name,
            bool(cfg["thinking_mode"]),
            int(cfg["max_tokens"]),
        )
        await asyncio.to_thread(
            write_json,
            output_root / variant_name / f"{item['idx']}.json",
            record,
        )
        print(f"finished variant={variant_name} idx={item['idx']} status={record['status']}")


def extract_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    for trace_item in record.get("trace", []):
        maybe = trace_item.get("content", {}).get("client_telemetry")
        if isinstance(maybe, dict):
            return maybe
    return {}


def build_summary(variant_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = VARIANTS[variant_name]
    items: List[Dict[str, Any]] = []
    for record in records:
        text = str(record.get("final_response", ""))
        meta = extract_meta(record)
        usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        items.append(
            {
                "idx": record.get("idx"),
                "status": record.get("status"),
                "final_marker": "FINAL_ANSWER:" in text.upper(),
                "finish_reason": meta.get("finish_reason"),
                "response_chars": len(text),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "http_attempt_count": meta.get("http_attempt_count"),
                "retry_count": meta.get("retry_count"),
                "total_latency_seconds": meta.get("total_latency_seconds"),
                "attempts": meta.get("attempts"),
            }
        )
    success = sum(i["status"] == "success" for i in items)
    markers = sum(bool(i["final_marker"]) for i in items)
    attempts = sum(int(i.get("http_attempt_count") or 0) for i in items)
    retries = sum(int(i.get("retry_count") or 0) for i in items)
    return {
        "variant": variant_name,
        "thinking_mode": cfg["thinking_mode"],
        "max_tokens": cfg["max_tokens"],
        "n": len(items),
        "success": success,
        "errors": len(items) - success,
        "final_marker_count": markers,
        "final_marker_rate": markers / len(items) if items else 0.0,
        "length_finish_count": sum(i.get("finish_reason") == "length" for i in items),
        "stop_finish_count": sum(i.get("finish_reason") == "stop" for i in items),
        "http_attempts": attempts,
        "retries": retries,
        "mean_http_attempts": attempts / len(items) if items else 0.0,
        "items": items,
    }


async def main_async(args: argparse.Namespace) -> None:
    inputs, gold = load_rows()
    selected_inputs = [inputs[i] for i in HARD_CASE_IDX]
    selected_gold = [gold[i] for i in HARD_CASE_IDX]

    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    bundle = output_root / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "gold.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected_gold),
        encoding="utf-8",
    )

    tasks: List[Tuple[str, Dict[str, Any]]] = [
        (variant_name, item)
        for variant_name in VARIANTS
        for item in selected_inputs
    ]
    random.Random(args.seed).shuffle(tasks)
    print("execution_order=", [(v, int(i["idx"])) for v, i in tasks])

    client = InternChatClient()
    semaphore = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(
        *[
            run_one(client, semaphore, item, variant_name, VARIANTS[variant_name], output_root)
            for variant_name, item in tasks
        ]
    )

    all_summaries: List[Dict[str, Any]] = []
    for variant_name in VARIANTS:
        records = [
            json.loads((output_root / variant_name / f"{idx}.json").read_text(encoding="utf-8"))
            for idx in HARD_CASE_IDX
        ]
        summary = build_summary(variant_name, records)
        write_json(summary_root / f"{variant_name}.json", summary)
        all_summaries.append(summary)

    write_json(
        summary_root / "comparison_raw.json",
        {
            "hard_cases": HARD_CASE_IDX,
            "seed": args.seed,
            "concurrency": args.concurrency,
            "variants": all_summaries,
        },
    )
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
