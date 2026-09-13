"""Run the reproducible ARC-Easy online serving benchmark."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkMethod:
    served_model_name: str
    result_directory_name: str


METHODS = {
    "baseline": BenchmarkMethod(
        served_model_name="qwen2.5-14b-instruct-fp16",
        result_directory_name="ARC-Easy-baseline-vllm-hust",
    ),
    "draft": BenchmarkMethod(
        served_model_name="qwen2.5-14b-draft-vspec-fp16",
        result_directory_name="ARC-Easy-draft-vspec",
    ),
    "eagle": BenchmarkMethod(
        served_model_name="qwen2.5-14b-eagle-vspec-fp16",
        result_directory_name="ARC-Easy-eagle-vspec",
    ),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def request_rate(value: str) -> str:
    if value == "inf":
        return value
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-vspec-bench",
        description="Run the vSpec ARC-Easy online benchmark protocol.",
    )
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--served-model-name")
    parser.add_argument("--tokenizer", default="/model/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/run_dir/materialized.jsonl"),
    )
    parser.add_argument("--output-len", type=positive_int, default=256)
    parser.add_argument("--num-prompts", type=positive_int, default=200)
    parser.add_argument("--request-rate", type=request_rate, default="inf")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--result-filename", default="ARC-Easy.json")
    parser.add_argument("--vllm-executable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to vllm bench serve (place after --).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    options = build_parser().parse_args(argv)
    method = METHODS[options.method]
    if not options.served_model_name:
        options.served_model_name = method.served_model_name
    if options.result_dir is None:
        options.result_dir = Path("/run_dir/benchmark_results") / method.result_directory_name
    if options.extra_args and options.extra_args[0] == "--":
        options.extra_args = options.extra_args[1:]
    return options


def resolve_vllm_executable(configured: str | None = None) -> str:
    environment_vllm = Path(sys.executable).with_name("vllm")
    executable = (
        configured
        or shutil.which("vllm")
        or (str(environment_vllm) if environment_vllm.is_file() else None)
    )
    if not executable:
        raise RuntimeError("vllm executable was not found in PATH")
    return executable


def build_benchmark_command(options: argparse.Namespace) -> list[str]:
    command = [
        resolve_vllm_executable(options.vllm_executable),
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        options.base_url,
        "--endpoint",
        options.endpoint,
        "--model",
        options.served_model_name,
        "--tokenizer",
        options.tokenizer,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(options.dataset_path),
        "--disable-shuffle",
        "--custom-output-len",
        str(options.output_len),
        "--num-prompts",
        str(options.num_prompts),
        "--request-rate",
        options.request_rate,
        "--temperature",
        f"{options.temperature:g}",
        "--seed",
        str(options.seed),
        "--save-result",
        "--result-dir",
        str(options.result_dir),
        "--result-filename",
        options.result_filename,
    ]
    command.extend(options.extra_args)
    return command


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_args(argv)
    command = build_benchmark_command(options)
    if options.dry_run:
        print(shlex.join(command))
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
