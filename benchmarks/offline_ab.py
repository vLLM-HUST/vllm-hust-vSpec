#!/usr/bin/env python3
"""Run a comparable fixed-gamma or vSpec Adaptive offline benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from vllm_hust_vspec.cli import generate_capture_sizes
from vllm_hust_vspec.config import PluginSettings

MODELS = {
    "draft": (
        "draft_model",
        "/data/shared-models/Qwen2.5-14B-Instruct",
        "/data/shared-models/Qwen2.5-0.5B-Instruct",
        "qwen25_14b",
    ),
    "eagle": (
        "eagle",
        "/data/shared-models/Qwen2.5-14B-Instruct",
        "/data/shared-models/Eagle-Qwen2.5-14B-Instruct",
        "qwen25_14b",
    ),
    "eagle3": (
        "eagle3",
        "/data/shared-models/Qwen3-8B",
        "/data/shared-models/qwen3_8b_eagle3",
        "qwen3_8b",
    ),
    "ngram": (
        "ngram",
        "/data/shared-models/Qwen2.5-14B-Instruct",
        "ngram",
        "qwen25_14b",
    ),
    "ngram_gpu": (
        "ngram_gpu",
        "/data/shared-models/Qwen2.5-14B-Instruct",
        "ngram_gpu",
        "qwen25_14b",
    ),
}
METHOD_CHOICES = (*MODELS, "dflash")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--target-model", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--manifest-family")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the Target model without speculative decoding.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=(
            "eager",
            "piecewise",
            "full-decode-only",
            "full",
            "full-and-piecewise",
        ),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gamma", type=int, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "sharegpt"), default="gsm8k")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--first-prompt-max-tokens",
        type=int,
        help="Use a shorter first request for batch-transition smoke tests.",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of visible NPU memory reserved by each vLLM instance.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--adaptive-profile", type=Path)
    parser.add_argument("--adaptive-policy", choices=("profile", "online"))
    parser.add_argument("--adaptive-min-gamma", type=int, default=1)
    parser.add_argument("--adaptive-min-observations", type=int, default=32)
    parser.add_argument("--adaptive-control-interval", type=int, default=4)
    parser.add_argument("--adaptive-hysteresis", type=float, default=0.05)
    parser.add_argument("--adaptive-max-gamma-step", type=int, default=1)
    parser.add_argument(
        "--adaptive-latency-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive-latency-ewma-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--adaptive-gamma0-mode",
        choices=("sticky", "sync"),
        default="sticky",
    )
    parser.add_argument("--adaptive-trace", action="store_true")
    parser.add_argument("--adaptive-full-graph", action="store_true")
    parser.add_argument("--adaptive-async", action="store_true")
    parser.add_argument("--adaptive-online-window", type=int, default=32)
    parser.add_argument("--adaptive-online-exploration", type=float, default=0.01)
    parser.add_argument("--adaptive-online-warmup-samples", type=int, default=1)
    parser.add_argument(
        "--adaptive-online-warmup-return",
        choices=("best", "incumbent"),
        default="best",
    )
    parser.add_argument("--adaptive-refill-batch", type=int, default=0)
    parser.add_argument("--adaptive-entropy-stop", action="store_true")
    parser.add_argument("--adaptive-entropy-topk", type=int, default=2)
    parser.add_argument("--adaptive-entropy-threshold", type=float, default=0.3)
    parser.add_argument("--adaptive-entropy-scale", type=float, default=0.15)
    parser.add_argument("--draft-active-vocab-size", type=int, default=0)
    parser.add_argument("--draft-active-vocab-ids", type=Path)
    parser.add_argument("--draft-target-active-vocab-ids", type=Path)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument(
        "--draft-lm-head-quantization",
        choices=("none", "w8a16", "w8a8"),
        default="none",
    )
    parser.add_argument("--component-stats-prefix", type=Path)
    parser.add_argument(
        "--capture-policy",
        choices=("auto", "exact", "power2", "steady"),
        default="auto",
        help="Override the graph capture bucket policy used by the benchmark.",
    )
    parser.add_argument("--capture-request-step", type=int)
    parser.add_argument(
        "--cudagraph-num-of-warmups",
        type=int,
        default=0,
        help="Number of warmup replays before capturing each graph size.",
    )
    parser.add_argument(
        "--extra-capture-size",
        action="append",
        type=int,
        default=[],
        help="Add an explicit FULL-graph token bucket (repeatable).",
    )
    parser.add_argument(
        "--static-kernel",
        action="store_true",
        help="Compile static Ascend kernels during graph capture.",
    )
    parser.add_argument(
        "--reduce-sample",
        action="store_true",
        help="Use Ascend's reduced-logit sampling path.",
    )
    parser.add_argument(
        "--weight-prefetch",
        choices=("all", "mlp", "attention"),
        help="Enable Ascend weight prefetch for selected dense-model modules.",
    )
    parser.add_argument(
        "--confidence-accept-margin",
        "--eagle-confidence-accept-margin",
        dest="confidence_accept_margin",
        type=float,
        help="Accept a Draft token when its Target logit gap is at most this value.",
    )
    parser.add_argument(
        "--confidence-accept-from-position",
        "--eagle-confidence-accept-from-position",
        dest="confidence_accept_from_position",
        type=int,
        default=0,
        help="Apply confidence acceptance from this zero-based draft position.",
    )
    parser.add_argument(
        "--confidence-accept-after-tokens",
        "--eagle-confidence-accept-after-tokens",
        dest="confidence_accept_after_tokens",
        type=int,
        default=0,
        help="Keep each request strict for this many generated tokens.",
    )
    parser.add_argument(
        "--confidence-protected-token-id",
        "--eagle-confidence-protected-token-id",
        dest="confidence_protected_token_id",
        action="append",
        type=int,
        default=[],
        help="Force strict verification when Target or Draft uses this token ID.",
    )
    parser.add_argument(
        "--eagle-tensor-seq-lens",
        action="store_true",
        help="Use tensor-backed FIA sequence lengths for stable EAGLE FULL graphs.",
    )
    parser.add_argument(
        "--eagle-parallel-graph-updates",
        type=int,
        default=0,
        help="Parallel Target FIA graph-task update streams (experimental).",
    )
    parser.add_argument(
        "--super-kernel",
        action="store_true",
        help="Enable super-kernel optimization inside compiled FX graphs.",
    )
    parser.add_argument(
        "--aclgraph-super-kernel",
        action="store_true",
        help="Optimize captured outer ACL Graphs as super kernels.",
    )
    parser.add_argument(
        "--aclgraph-super-kernel-scope",
        choices=("all", "target", "draft"),
        default="all",
        help="Select which captured ACL Graph wrappers to optimize.",
    )
    parser.add_argument(
        "--adaptive-fia",
        action="store_true",
        help="Enable the batch-invariant adaptive FIA pipeline.",
    )
    parser.add_argument(
        "--adaptive-fia-min-batch",
        type=int,
        default=32,
        help="Minimum active batch size for adaptive FIA dispatch.",
    )
    parser.add_argument("--include-output-token-ids", action="store_true")
    return parser.parse_args()


def configure_plugin(args: argparse.Namespace) -> tuple[str, str, str, str]:
    if args.method == "dflash":
        if args.target_model is None or args.draft_model is None or not args.manifest_family:
            raise SystemExit("dflash requires --target-model, --draft-model, and --manifest-family")
        method = "dflash"
        target = str(args.target_model)
        draft = str(args.draft_model)
        family = args.manifest_family
    else:
        method, target, draft, family = MODELS[args.method]
        target = str(args.target_model) if args.target_model else target
        draft = str(args.draft_model) if args.draft_model else draft
        family = args.manifest_family or family
    adaptive_policy = getattr(args, "adaptive_policy", None)
    if adaptive_policy is None and args.adaptive_profile is not None:
        adaptive_policy = "profile"
    adaptive = adaptive_policy is not None
    draft_active_vocab_size = getattr(args, "draft_active_vocab_size", 0)
    draft_active_vocab_ids = getattr(args, "draft_active_vocab_ids", None)
    draft_target_active_vocab_ids = getattr(
        args,
        "draft_target_active_vocab_ids",
        None,
    )
    draft_lm_head_quantization = getattr(
        args,
        "draft_lm_head_quantization",
        "none",
    )
    draft_active_vocab = (
        draft_active_vocab_size > 0
        or draft_active_vocab_ids is not None
        or draft_lm_head_quantization != "none"
    )
    if (
        draft_active_vocab or draft_target_active_vocab_ids is not None
    ) and method != "draft_model":
        raise SystemExit("Draft active vocabulary requires --method draft")
    if draft_active_vocab_size and draft_active_vocab_ids is not None:
        raise SystemExit(
            "--draft-active-vocab-size and --draft-active-vocab-ids are mutually exclusive"
        )
    plugin_method = method if method not in {"ngram", "ngram_gpu"} else "draft_model"
    settings = PluginSettings(
        enabled=not args.baseline and method not in {"ngram", "ngram_gpu"},
        method=plugin_method,
        assume_shared_tokenizer=method == "draft_model",
        use_merged_full=(
            method == "draft_model"
            and (not adaptive or args.adaptive_full_graph)
            and args.execution_mode
            in {"full", "full-decode-only", "full-and-piecewise"}
        ),
        max_num_seqs=args.batch_size,
        draft_active_vocab=draft_active_vocab,
        draft_target_active_vocab=(draft_target_active_vocab_ids is not None),
        confidence_accept_margin=getattr(args, "confidence_accept_margin", None),
        confidence_accept_from_position=getattr(
            args,
            "confidence_accept_from_position",
            0,
        ),
        confidence_accept_after_tokens=getattr(
            args,
            "confidence_accept_after_tokens",
            0,
        ),
        confidence_protected_token_ids=tuple(getattr(args, "confidence_protected_token_id", ())),
        adaptive_speculation=adaptive,
        adaptive_policy=adaptive_policy or "profile",
        adaptive_profile_path=(
            str(args.adaptive_profile.resolve()) if args.adaptive_profile is not None else ""
        ),
        adaptive_max_gamma=max(args.gamma, 1),
        adaptive_min_gamma=(args.adaptive_min_gamma if not args.baseline else 1),
        adaptive_control_interval=args.adaptive_control_interval,
        adaptive_hysteresis=args.adaptive_hysteresis,
        adaptive_min_observations=args.adaptive_min_observations,
        adaptive_max_gamma_step=args.adaptive_max_gamma_step,
        adaptive_latency_calibration=args.adaptive_latency_calibration,
        adaptive_latency_ewma_weight=(args.adaptive_latency_ewma_weight),
        adaptive_gamma0_mode=args.adaptive_gamma0_mode,
        adaptive_trace=args.adaptive_trace,
        adaptive_full_graph=args.adaptive_full_graph,
        adaptive_async=args.adaptive_async,
        adaptive_online_window=getattr(args, "adaptive_online_window", 32),
        adaptive_online_exploration=getattr(
            args,
            "adaptive_online_exploration",
            0.01,
        ),
        adaptive_online_warmup_samples=getattr(
            args,
            "adaptive_online_warmup_samples",
            1,
        ),
        adaptive_online_warmup_return=getattr(
            args,
            "adaptive_online_warmup_return",
            "best",
        ),
        adaptive_refill_batch=getattr(args, "adaptive_refill_batch", 0),
        adaptive_entropy_stop=getattr(args, "adaptive_entropy_stop", False),
        adaptive_entropy_topk=getattr(args, "adaptive_entropy_topk", 2),
        adaptive_entropy_threshold=getattr(
            args,
            "adaptive_entropy_threshold",
            0.3,
        ),
        adaptive_entropy_scale=getattr(args, "adaptive_entropy_scale", 0.15),
    )
    os.environ.update(settings.as_environment())
    if getattr(args, "eagle_tensor_seq_lens", False):
        if method != "eagle":
            raise SystemExit("--eagle-tensor-seq-lens requires --method eagle")
        os.environ["VSPEC_EAGLE_TENSOR_SEQ_LENS"] = "1"
    parallel_graph_updates = getattr(args, "eagle_parallel_graph_updates", 0)
    if parallel_graph_updates:
        if method != "eagle":
            raise SystemExit("--eagle-parallel-graph-updates requires --method eagle")
        if parallel_graph_updates < 2:
            raise SystemExit("--eagle-parallel-graph-updates must be at least 2")
        os.environ["VSPEC_EAGLE_PARALLEL_GRAPH_UPDATES"] = str(parallel_graph_updates)
    if draft_active_vocab_size > 0:
        os.environ["VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_SIZE"] = str(draft_active_vocab_size)
    if draft_active_vocab_ids is not None:
        os.environ["VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_IDS_PATH"] = str(
            draft_active_vocab_ids.resolve()
        )
    if draft_target_active_vocab_ids is not None:
        os.environ["VLLM_ASCEND_DRAFT_TARGET_ACTIVE_VOCAB_IDS_PATH"] = str(
            draft_target_active_vocab_ids.resolve()
        )
    if draft_lm_head_quantization != "none":
        suffix = draft_lm_head_quantization.upper()
        os.environ[f"VLLM_ASCEND_DRAFT_LM_HEAD_{suffix}"] = "1"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if args.component_stats_prefix is not None:
        stats_prefix = args.component_stats_prefix.resolve()
        stats_prefix.parent.mkdir(parents=True, exist_ok=True)
        os.environ["TREE_DRAFT_STATS_PREFIX"] = str(stats_prefix)
        os.environ["PEARL_STAGE5_WORKER_ROLE"] = "target"
        os.environ["PEARL_STAGE5_TARGET_COMPONENT_PERF"] = "1"
        os.environ["PEARL_STAGE5_TARGET_COMPONENT_PERF_INTERVAL"] = "0"
    if args.device is not None:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    allowed_plugins = [name for name in os.environ.get("VLLM_PLUGINS", "").split(",") if name]
    for name in ("ascend", "vspec"):
        if name not in allowed_plugins:
            allowed_plugins.append(name)
    os.environ["VLLM_PLUGINS"] = ",".join(allowed_plugins)
    return method, target, draft, family


def output_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


NUMBER_PATTERN = r"-?\d[\d,]*(?:\.\d+)?"


def normalize_number(value: str) -> str:
    value = value.replace(",", "").strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def extract_gsm8k_answer(text: str) -> str | None:
    patterns = (
        rf"####\s*({NUMBER_PATTERN})",
        rf"\\boxed\{{\s*({NUMBER_PATTERN})\s*\}}",
        rf"(?:final\s+)?answer\s+is\s*[:=]?\s*({NUMBER_PATTERN})",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])
    matches = re.findall(NUMBER_PATTERN, text)
    return normalize_number(matches[-1]) if matches else None


def evaluate_gsm8k(
    dataset_path: str,
    prompts: list[str],
    generated_texts: list[str],
) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(dataset_path, columns=["question", "answer"])
    references = dict(
        zip(
            table.column("question").to_pylist(),
            table.column("answer").to_pylist(),
            strict=True,
        )
    )
    predictions: list[str | None] = []
    expected: list[str | None] = []
    for prompt, generated_text in zip(prompts, generated_texts, strict=True):
        if prompt not in references:
            raise ValueError("GSM8K manifest prompt was not found in the dataset")
        predictions.append(extract_gsm8k_answer(generated_text))
        expected.append(extract_gsm8k_answer(references[prompt]))
    correct = sum(
        prediction is not None and prediction == reference
        for prediction, reference in zip(predictions, expected, strict=True)
    )
    return {
        "gsm8k_correct": correct,
        "gsm8k_accuracy": correct / len(prompts),
        "gsm8k_extracted_answers": sum(answer is not None for answer in predictions),
    }


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.num_prompts, args.max_tokens) <= 0 or args.gamma < 0:
        raise SystemExit(
            "; ".join(
                (
                    "batch, prompt count, and max tokens must be positive",
                    "gamma must be nonnegative",
                )
            )
        )
    if min(args.tensor_parallel_size, args.draft_tensor_parallel_size) <= 0:
        raise SystemExit("tensor parallel sizes must be positive")
    if args.cudagraph_num_of_warmups < 0:
        raise SystemExit("cudagraph warmup count must be nonnegative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("gpu memory utilization must be in (0, 1]")
    if args.baseline != (args.gamma == 0):
        raise SystemExit("--baseline requires --gamma 0, and gamma 0 requires --baseline")
    if args.adaptive_fia_min_batch <= 0:
        raise SystemExit("adaptive FIA minimum batch size must be positive")
    adaptive_policy = args.adaptive_policy
    if adaptive_policy is None and args.adaptive_profile is not None:
        adaptive_policy = "profile"
    if adaptive_policy is not None:
        if adaptive_policy == "profile" and args.adaptive_profile is None:
            raise SystemExit("profile adaptive policy requires --adaptive-profile")
        if adaptive_policy == "online" and args.adaptive_profile is not None:
            raise SystemExit("online adaptive policy does not use --adaptive-profile")
        if not 0 <= args.adaptive_min_gamma <= args.gamma:
            raise SystemExit("adaptive min gamma must be in [0, gamma]")
        if args.async_scheduling and not args.adaptive_async:
            raise SystemExit("current Adaptive runtime requires --no-async-scheduling")
        if args.execution_mode not in {"eager", "piecewise"} and not args.adaptive_full_graph:
            raise SystemExit("current Adaptive runtime requires eager or piecewise")

    method, target, draft, family = configure_plugin(args)

    # Import after exporting plugin settings so registration happens with the
    # same contract in this process and spawned EngineCore workers.
    from vllm import LLM, SamplingParams

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["family"] != family:
        raise SystemExit(f"manifest family {manifest['family']} does not match {family}")
    dataset = manifest["datasets"][args.dataset]
    prompts = dataset["prompts"][: args.num_prompts]
    prompt_lengths = dataset["prompt_lengths"][: args.num_prompts]

    enforce_eager = args.execution_mode == "eager"
    llm_args: dict[str, Any] = {
        "model": target,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "disable_log_stats": True,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": (
            args.max_num_batched_tokens
            if args.max_num_batched_tokens is not None
            else (
                32768 + args.batch_size * args.gamma
                if method == "draft_model"
                else max(8192, args.batch_size * (args.gamma + 1))
            )
        ),
        "max_model_len": args.max_model_len,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": args.prefix_caching,
        "async_scheduling": args.async_scheduling,
        "seed": 0,
        "dtype": getattr(args, "dtype", "auto"),
    }
    if not args.baseline:
        speculative_config: dict[str, Any] = {
            "method": method,
            "model": draft,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
            "num_speculative_tokens": args.gamma,
            "enforce_eager": enforce_eager,
        }
        if method == "draft_model":
            speculative_config["use_heterogeneous_vocab"] = True
        elif method in {"ngram", "ngram_gpu"}:
            speculative_config["prompt_lookup_min"] = args.prompt_lookup_min
            speculative_config["prompt_lookup_max"] = args.prompt_lookup_max
        llm_args["speculative_config"] = speculative_config
    if not enforce_eager:
        graph_mode = args.execution_mode.replace("-", "_").upper()
        capture_sizes = generate_capture_sizes(
            args.batch_size,
            args.gamma,
            method,
            policy=(
                getattr(args, "capture_policy", "auto")
                if getattr(args, "capture_policy", "auto") != "auto"
                else ("exact" if method in {"eagle", "eagle3"} else "auto")
            ),
            dynamic_widths=(adaptive_policy is not None and not args.baseline),
            request_step=getattr(args, "capture_request_step", None),
        )
        if any(size <= 0 for size in args.extra_capture_size):
            raise SystemExit("extra capture sizes must be positive")
        capture_sizes = sorted(set(capture_sizes).union(args.extra_capture_size))
        llm_args["compilation_config"] = {
            "mode": 3,
            "cudagraph_mode": graph_mode,
            "cudagraph_capture_sizes": capture_sizes,
            "cudagraph_num_of_warmups": args.cudagraph_num_of_warmups,
        }
        if args.static_kernel:
            additional_config = llm_args.setdefault("additional_config", {})
            additional_config["ascend_compilation_config"] = {
                "enable_npugraph_ex": True,
                "enable_static_kernel": args.static_kernel,
            }

    if args.adaptive_fia:
        additional_config = llm_args.setdefault("additional_config", {})
        additional_config["adaptive_fia_pipeline_config"] = {
            "enabled": True,
            "mode": "auto",
            "three_stage_max_kv_tokens": 4096,
            "three_stage_max_working_set_bytes": 32 * 1024 * 1024,
            "auto_min_batch_size": args.adaptive_fia_min_batch,
            "enable_two_stage": True,
            "enable_three_stage": True,
            "enable_graph_execution": not enforce_eager,
            "enable_graph_task_update": args.execution_mode == "full",
            "enable_preallocated_out": True,
            "collect_stats": False,
        }
    if args.reduce_sample:
        additional_config = llm_args.setdefault("additional_config", {})
        additional_config["enable_reduce_sample"] = True
    if args.weight_prefetch is not None:
        prefetch_ratio = {
            "attn": {
                "qkv": 1.0 if args.weight_prefetch in {"all", "attention"} else 0.0,
                "o": 1.0 if args.weight_prefetch in {"all", "attention"} else 0.0,
            },
            "mlp": {
                "gate_up": 1.0 if args.weight_prefetch in {"all", "mlp"} else 0.0,
                "down": 1.0 if args.weight_prefetch in {"all", "mlp"} else 0.0,
            },
        }
        additional_config = llm_args.setdefault("additional_config", {})
        additional_config["weight_prefetch_config"] = {
            "enabled": True,
            "prefetch_ratio": prefetch_ratio,
        }

    print("VSPEC_BENCH_CONFIG:", json.dumps(llm_args, sort_keys=True), flush=True)
    llm = LLM(**llm_args)
    sampling_params: SamplingParams | list[SamplingParams] = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_tokens=args.max_tokens,
    )
    if args.first_prompt_max_tokens is not None:
        if not 0 < args.first_prompt_max_tokens <= args.max_tokens:
            raise SystemExit("first-prompt-max-tokens must be in [1, max-tokens]")
        sampling_params = [
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                seed=0,
                max_tokens=(args.first_prompt_max_tokens if index == 0 else args.max_tokens),
            )
            for index in range(len(prompts))
        ]
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    rows = [list(output.outputs[0].token_ids) for output in outputs]
    generated_texts = [output.outputs[0].text for output in outputs]
    output_tokens = sum(map(len, rows))
    result: dict[str, Any] = {
        "method": args.method,
        "target_model": target,
        "draft_model": draft if not args.baseline else None,
        "manifest_family": family,
        "baseline": args.baseline,
        "adaptive": adaptive_policy is not None,
        "adaptive_policy": adaptive_policy,
        "adaptive_profile": (
            str(args.adaptive_profile.resolve()) if args.adaptive_profile is not None else None
        ),
        "adaptive_gamma0_mode": (
            args.adaptive_gamma0_mode if adaptive_policy is not None else None
        ),
        "adaptive_latency_calibration": (
            args.adaptive_latency_calibration if adaptive_policy is not None else None
        ),
        "adaptive_min_gamma": (args.adaptive_min_gamma if adaptive_policy is not None else None),
        "adaptive_min_observations": (
            args.adaptive_min_observations if adaptive_policy is not None else None
        ),
        "adaptive_control_interval": (
            args.adaptive_control_interval if adaptive_policy is not None else None
        ),
        "adaptive_hysteresis": (args.adaptive_hysteresis if adaptive_policy is not None else None),
        "adaptive_max_gamma_step": (
            args.adaptive_max_gamma_step if adaptive_policy is not None else None
        ),
        "adaptive_latency_ewma_weight": (
            args.adaptive_latency_ewma_weight if adaptive_policy is not None else None
        ),
        "adaptive_online_window": (
            args.adaptive_online_window if adaptive_policy == "online" else None
        ),
        "adaptive_online_exploration": (
            args.adaptive_online_exploration if adaptive_policy == "online" else None
        ),
        "adaptive_online_warmup_samples": (
            args.adaptive_online_warmup_samples if adaptive_policy == "online" else None
        ),
        "adaptive_online_warmup_return": (
            args.adaptive_online_warmup_return if adaptive_policy == "online" else None
        ),
        "adaptive_entropy_stop": (
            args.adaptive_entropy_stop if adaptive_policy == "online" else None
        ),
        "adaptive_entropy_threshold": (
            args.adaptive_entropy_threshold if adaptive_policy == "online" else None
        ),
        "adaptive_entropy_scale": (
            args.adaptive_entropy_scale if adaptive_policy == "online" else None
        ),
        "execution_mode": args.execution_mode,
        "cudagraph_num_of_warmups": args.cudagraph_num_of_warmups,
        "reduce_sample": args.reduce_sample,
        "weight_prefetch": args.weight_prefetch,
        "confidence_accept_margin": args.confidence_accept_margin,
        "confidence_accept_from_position": args.confidence_accept_from_position,
        "confidence_accept_after_tokens": args.confidence_accept_after_tokens,
        "confidence_protected_token_ids": args.confidence_protected_token_id,
        "adaptive_refill_batch": args.adaptive_refill_batch,
        "eagle_tensor_seq_lens": args.eagle_tensor_seq_lens,
        "eagle_parallel_graph_updates": args.eagle_parallel_graph_updates,
        "static_kernel": args.static_kernel,
        "super_kernel": args.super_kernel,
        "aclgraph_super_kernel": args.aclgraph_super_kernel,
        "aclgraph_super_kernel_scope": args.aclgraph_super_kernel_scope,
        "adaptive_fia": args.adaptive_fia,
        "adaptive_fia_min_batch": args.adaptive_fia_min_batch,
        "async_scheduling": args.async_scheduling,
        "prefix_caching": args.prefix_caching,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "dataset": args.dataset,
        "num_prompts": len(rows),
        "max_tokens": args.max_tokens,
        "first_prompt_max_tokens": args.first_prompt_max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "draft_tensor_parallel_size": (
            args.draft_tensor_parallel_size if not args.baseline else None
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": getattr(args, "dtype", "auto"),
        "max_num_batched_tokens": llm_args["max_num_batched_tokens"],
        "elapsed_s": elapsed,
        "input_tokens": sum(prompt_lengths),
        "prompt_lengths": prompt_lengths,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed,
        "row_output_hashes": [output_hash(row) for row in rows],
        "output_lengths": list(map(len, rows)),
    }
    if args.include_output_token_ids:
        result["row_output_token_ids"] = rows
    if args.dataset == "gsm8k":
        result.update(evaluate_gsm8k(dataset["path"], prompts, generated_texts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("VSPEC_BENCH_RESULT:", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
