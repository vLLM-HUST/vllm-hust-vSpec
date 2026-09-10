"""Run fixed-gamma measurements and build a vSpec Adaptive profile."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .fitting import fit_profile


def _int_list(value: str, name: str, *, minimum: int) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from exc
    if not values or values[0] < minimum:
        raise argparse.ArgumentTypeError(f"{name} values must be >= {minimum}")
    return values


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _component_bucket(stats: Mapping[str, Any], gamma: int) -> Mapping[str, Any]:
    target = stats.get("target")
    if not isinstance(target, dict):
        raise ValueError("component stats require a target object")
    preferred = ("spec", "steady_spec") if gamma > 0 else ("decode",)
    for name in preferred:
        bucket = target.get(name)
        if isinstance(bucket, dict) and int(bucket.get("calls", 0)) > 0:
            return bucket
    raise ValueError(
        f"component stats do not contain a usable {'spec' if gamma else 'decode'} bucket"
    )


def _estimated_context_tokens(
    benchmark: Mapping[str, Any],
    average_batch_size: float,
) -> int:
    prompt_lengths = benchmark.get("prompt_lengths")
    output_lengths = benchmark.get("output_lengths")
    if not isinstance(prompt_lengths, list) or not isinstance(output_lengths, list):
        raise ValueError("benchmark result requires prompt_lengths and output_lengths")
    if len(prompt_lengths) != len(output_lengths) or not prompt_lengths:
        raise ValueError("prompt_lengths and output_lengths must be non-empty and aligned")

    weights = [max(int(length), 1) for length in output_lengths]
    weighted_context = sum(
        weight * (int(prompt) + max(int(output) - 1, 0) / 2)
        for prompt, output, weight in zip(
            prompt_lengths,
            output_lengths,
            weights,
            strict=True,
        )
    ) / sum(weights)
    return max(0, round(average_batch_size * weighted_context))


def measurement_to_samples(
    benchmark: Mapping[str, Any],
    component_stats: Mapping[str, Any],
    gamma: int,
    *,
    parallel_drafting: bool = False,
) -> tuple[dict[str, float | int], dict[str, float | int] | None]:
    """Convert one fixed-gamma run into Target and Draft fit samples."""
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    bucket = _component_bucket(component_stats, gamma)
    calls = int(bucket.get("calls", 0))
    reqs = int(bucket.get("reqs", 0))
    tokens = int(bucket.get("tokens", 0))
    if calls <= 0 or reqs <= 0 or tokens <= 0:
        raise ValueError("component stats calls, reqs, and tokens must be positive")

    average_batch = reqs / calls
    context_tokens = _estimated_context_tokens(benchmark, average_batch)
    target_latency_sum = sum(
        float(bucket.get(name, 0.0))
        for name in ("forward_npu_ms", "logits_npu_ms", "sampler_npu_ms")
    )
    if not math.isfinite(target_latency_sum) or target_latency_sum <= 0:
        raise ValueError("component stats contain no positive Target latency")
    target_sample: dict[str, float | int] = {
        "context_tokens": context_tokens,
        "batched_tokens": max(1, round(tokens / calls)),
        "latency_ms": target_latency_sum / calls,
    }
    if gamma == 0:
        return target_sample, None

    graph_draft_sum = max(
        float(bucket.get("graph_replay_npu_ms", 0.0)) - float(bucket.get("forward_npu_ms", 0.0)),
        0.0,
    )
    residual_draft_sum = max(
        float(bucket.get("total_cpu_ms", 0.0))
        - float(bucket.get("prepare_cpu_ms", 0.0))
        - target_latency_sum
        - float(bucket.get("sampler_cpu_ms", 0.0))
        - float(bucket.get("bookkeeping_cpu_ms", 0.0)),
        0.0,
    )
    draft_latency_sum = graph_draft_sum or residual_draft_sum
    if not math.isfinite(draft_latency_sum) or draft_latency_sum <= 0:
        raise ValueError("component stats contain no positive Draft latency")
    draft_sample: dict[str, float | int]
    if parallel_drafting:
        draft_sample = {
            "context_tokens": context_tokens,
            "batched_tokens": max(
                1,
                round(average_batch * (gamma + 1)),
            ),
            "latency_ms": draft_latency_sum / calls,
        }
    else:
        draft_sample = {
            "context_tokens": context_tokens + round((gamma - 1) * average_batch / 2),
            "batched_tokens": max(1, round(average_batch)),
            "latency_ms": draft_latency_sum / (calls * gamma),
        }
    return target_sample, draft_sample


def build_profile_from_measurements(
    measurements: Sequence[Mapping[str, Any]],
    *,
    max_gamma: int,
    default_acceptance_rate: float,
    parallel_drafting: bool = False,
) -> dict[str, Any]:
    target_samples: list[Mapping[str, Any]] = []
    draft_samples: list[Mapping[str, Any]] = []
    throughput_by_batch_gamma: dict[tuple[int, int], list[float]] = defaultdict(list)
    for measurement in measurements:
        gamma = int(measurement["gamma"])
        batch_size = int(measurement["batch_size"])
        benchmark = measurement["benchmark"]
        stats = measurement["component_stats"]
        if not isinstance(benchmark, dict) or not isinstance(stats, dict):
            raise ValueError("measurement benchmark and component_stats must be objects")
        target, draft = measurement_to_samples(
            benchmark,
            stats,
            gamma,
            parallel_drafting=parallel_drafting,
        )
        target_samples.append(target)
        if draft is not None:
            draft_samples.append(draft)
        throughput_by_batch_gamma[(batch_size, gamma)].append(
            float(benchmark["output_tokens_per_s"])
        )

    policy: dict[str, int] = {}
    batch_sizes = sorted({batch for batch, _ in throughput_by_batch_gamma})
    for batch_size in batch_sizes:
        candidates = [
            (
                statistics.fmean(values),
                gamma,
            )
            for (batch, gamma), values in throughput_by_batch_gamma.items()
            if batch == batch_size
        ]
        _, best_gamma = max(candidates, key=lambda item: (item[0], -item[1]))
        policy[str(batch_size)] = best_gamma

    document = {
        "max_speculative_tokens": max_gamma,
        "initial_gamma": policy[str(batch_sizes[0])],
        "default_acceptance_rate": default_acceptance_rate,
        "target_samples": target_samples,
        "draft_samples": draft_samples,
        "draft_parallel": parallel_drafting,
        "batch_gamma_policy": policy,
    }
    return fit_profile(document)


def _run_measurement(
    options: argparse.Namespace,
    batch_size: int,
    gamma: int,
    max_tokens: int,
) -> dict[str, Any]:
    stem = f"{options.method}_{options.execution_mode}_b{batch_size}_g{gamma}_m{max_tokens}"
    result_path = options.output_dir / f"{stem}.json"
    stats_prefix = options.output_dir / f"{stem}.components"
    stats_path = Path(f"{stats_prefix}.target.json")
    log_path = options.output_dir / f"{stem}.log"
    for path in (result_path, stats_path, log_path):
        path.unlink(missing_ok=True)

    benchmark_script = Path(__file__).resolve().parents[3] / "benchmarks" / "offline_ab.py"
    command = [
        sys.executable,
        str(benchmark_script),
        "--method",
        options.method,
        "--execution-mode",
        options.execution_mode,
        "--batch-size",
        str(batch_size),
        "--gamma",
        str(gamma),
        "--dataset",
        options.dataset,
        "--manifest",
        str(options.manifest),
        "--num-prompts",
        str(options.num_prompts),
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(options.max_model_len),
        "--max-num-batched-tokens",
        str(options.max_num_batched_tokens),
        "--output",
        str(result_path),
        "--component-stats-prefix",
        str(stats_prefix),
        ("--async-scheduling" if options.async_scheduling else "--no-async-scheduling"),
    ]
    if gamma == 0:
        command.append("--baseline")
    command.append("--prefix-caching" if options.prefix_caching else "--no-prefix-caching")
    if options.device is not None:
        command.extend(["--device", options.device])
    if options.target_model is not None:
        command.extend(["--target-model", str(options.target_model)])
    if options.draft_model is not None:
        command.extend(["--draft-model", str(options.draft_model)])
    if options.manifest_family is not None:
        command.extend(["--manifest-family", options.manifest_family])
    print(f"Profiling batch={batch_size} gamma={gamma} max_tokens={max_tokens}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"measurement failed with exit code {completed.returncode}; see {log_path}"
        )
    benchmark = _load_json_object(result_path)
    component_stats = _load_json_object(stats_path)
    return {
        "batch_size": batch_size,
        "gamma": gamma,
        "max_tokens": max_tokens,
        "result_path": str(result_path),
        "component_stats_path": str(stats_path),
        "log_path": str(log_path),
        "benchmark": benchmark,
        "component_stats": component_stats,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-vspec-profile",
        description="Collect fixed-gamma NPU measurements and fit an adaptive profile.",
    )
    parser.add_argument(
        "--method",
        choices=("draft", "eagle", "eagle3", "dflash"),
        required=True,
    )
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "piecewise", "full-decode-only", "full"),
        required=True,
    )
    parser.add_argument("--batches", default="8,32,64,128")
    parser.add_argument("--gammas", default="0,1,2,3,4")
    parser.add_argument("--max-tokens-grid", default="128,512")
    parser.add_argument("--dataset", choices=("gsm8k", "sharegpt"), default="gsm8k")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--manifest-family")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--default-acceptance-rate", type=float, default=0.7)
    parser.add_argument("--device")
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    options = parser.parse_args(argv)
    try:
        batches = _int_list(options.batches, "batches", minimum=1)
        gammas = _int_list(options.gammas, "gammas", minimum=0)
        max_tokens_grid = _int_list(
            options.max_tokens_grid,
            "max-tokens-grid",
            minimum=1,
        )
        if max(gammas) == 0:
            raise ValueError("gammas must include at least one positive value")
        if options.num_prompts <= 0 or options.max_model_len <= 0:
            raise ValueError("num-prompts and max-model-len must be positive")
        if options.method == "dflash" and (
            options.target_model is None
            or options.draft_model is None
            or not options.manifest_family
        ):
            raise ValueError(
                "dflash profiling requires --target-model, --draft-model, and --manifest-family"
            )
        if not 0 <= options.default_acceptance_rate <= 1:
            raise ValueError("default-acceptance-rate must be in [0, 1]")
        options.output_dir.mkdir(parents=True, exist_ok=True)
        if options.max_num_batched_tokens is None:
            options.max_num_batched_tokens = (
                32768 + max(batches) * max(gammas)
                if options.method == "draft"
                else max(8192, max(batches) * (max(gammas) + 1))
            )
        if options.max_num_batched_tokens <= 0:
            raise ValueError("max-num-batched-tokens must be positive")
        measurements = [
            _run_measurement(options, batch_size, gamma, max_tokens)
            for max_tokens in max_tokens_grid
            for batch_size in batches
            for gamma in gammas
        ]
        profile = build_profile_from_measurements(
            measurements,
            max_gamma=max(gammas),
            default_acceptance_rate=options.default_acceptance_rate,
            parallel_drafting=options.method == "dflash",
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    manifest_path = options.output_dir / "measurements.json"
    manifest_path.write_text(
        json.dumps({"measurements": measurements}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    options.output_profile.parent.mkdir(parents=True, exist_ok=True)
    options.output_profile.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote profile: {options.output_profile}", flush=True)
    print(f"Wrote measurements: {manifest_path}", flush=True)
