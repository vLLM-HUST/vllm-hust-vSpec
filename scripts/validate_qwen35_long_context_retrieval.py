#!/usr/bin/env python3
"""Run a deterministic 26-case long-context retrieval gate."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from urllib import error, request

from transformers import AutoTokenizer

TARGET_LENGTHS = (
    8192,
    16384,
    32768,
    49152,
    65536,
    81920,
    98304,
    114688,
    131072,
    163840,
    196608,
    229376,
    250000,
)
POSITIONS = (0.1, 0.9)
FILLER = "Archive record: the requested verification value is not present in this record.\n"


def chat_token_count(tokenizer: object, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return len(input_ids)


def post_json(url: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def build_messages(
    tokenizer: object,
    target_tokens: int,
    position: float,
    case_id: int,
) -> tuple[list[dict[str, str]], str, int]:
    needle = f"VSPEC-LONG-CONTEXT-{case_id:02d}-A7F3"
    marker = f"NEEDLE-{case_id:02d}"
    prefix = (
        f"Search the archive for record {marker}. Return only its VALUE, with no explanation.\n"
        "ARCHIVE START\n"
    )
    record = f"Record {marker}; VALUE={needle}\n"
    suffix = f"ARCHIVE END\nWhat is the VALUE for record {marker}?"
    system = "You are a deterministic retrieval engine. Follow the requested output format exactly."

    fixed_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prefix + record + suffix},
    ]
    fixed_tokens = chat_token_count(tokenizer, fixed_messages)
    filler_tokens = max(0, target_tokens - fixed_tokens)
    filler_piece_tokens = max(1, len(tokenizer.encode(FILLER, add_special_tokens=False)))
    repeats = filler_tokens // filler_piece_tokens
    left_repeats = int(repeats * position)
    right_repeats = repeats - left_repeats
    content = prefix + FILLER * left_repeats + record + FILLER * right_repeats + suffix
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    actual_tokens = chat_token_count(tokenizer, messages)
    return messages, needle, actual_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18185")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    results: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for case_id, (target_tokens, position) in enumerate(
        ((length, position) for length in TARGET_LENGTHS for position in POSITIONS),
        start=1,
    ):
        messages, needle, actual_tokens = build_messages(
            tokenizer,
            target_tokens,
            position,
            case_id,
        )
        started = time.monotonic()
        response = post_json(
            endpoint,
            {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            args.timeout,
        )
        elapsed = time.monotonic() - started
        text = str(response["choices"][0]["message"]["content"]).strip()
        passed = needle in text
        result = {
            "case": case_id,
            "target_tokens": target_tokens,
            "actual_prompt_tokens": actual_tokens,
            "needle_position": position,
            "expected": needle,
            "response": text,
            "passed": passed,
            "elapsed_seconds": elapsed,
            "usage": response.get("usage"),
        }
        results.append(result)
        args.output.write_text(
            json.dumps({"completed": len(results), "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"case={case_id:02d}/26 tokens={actual_tokens} position={position:.1f} "
            f"passed={passed} elapsed={elapsed:.2f}s",
            flush=True,
        )

    passed_count = sum(bool(item["passed"]) for item in results)
    report = {
        "model": args.model,
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
