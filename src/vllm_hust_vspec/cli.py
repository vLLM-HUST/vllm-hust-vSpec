"""Configuration-driven launcher for the vSpec vLLM plugin."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tomllib
from collections.abc import Mapping, Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from .config import METHOD_ALIASES, PluginSettings
from .model_store import ModelStoreError, resolve_default_model

PLUGIN_ENTRY_POINT = "vspec"
ASCEND_PLATFORM_PLUGIN = "ascend"

SERVE_PROTOCOLS: dict[str, dict[str, Any]] = {
    "arc-easy": {
        "target_model": "/model/Qwen2.5-14B-Instruct",
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "max_model_len": 32768,
        "dtype": "float16",
        "block_size": 128,
        "tensor_parallel_size": 1,
        "draft_tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "host": "127.0.0.1",
        "port": 18180,
        "graph_mode": "full-decode-only",
        "generation_config": "auto",
        "async_scheduling": False,
        "prefix_caching": False,
        "chunked_prefill": True,
        "capture_policy": "auto",
        "disable_log_stats": False,
    }
}

ARC_EASY_SERVED_MODEL_NAMES = {
    "draft_model": "qwen2.5-14b-draft-vspec-fp16",
    "eagle": "qwen2.5-14b-eagle-vspec-fp16",
}

EAGLE_REQUEST_CAPTURE_SIZES = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
)

EAGLE_ENVIRONMENT_NAMES = (
    "VLLM_ASCEND_EAGLE_TREE_WIDTH",
    "VLLM_ASCEND_EAGLE_DRAFT_ACTIVE_VOCAB_SIZE",
    "VLLM_ASCEND_EAGLE_DRAFT_ACTIVE_VOCAB_IDS_PATH",
    "VLLM_ASCEND_EAGLE_TARGET_ACTIVE_VOCAB_IDS_PATH",
    "VLLM_ASCEND_EAGLE_DRAFT_LM_HEAD_W8A16",
    "VLLM_ASCEND_EAGLE_DRAFT_LM_HEAD_W8A8",
    "VLLM_ASCEND_EAGLE_DISABLE_DRAFT_TORCH_COMPILE",
    "VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_TOPK",
    "VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_AFTER_TOKENS",
    "VLLM_ASCEND_EAGLE_SPEC_METADATA_CACHE",
    "VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL",
    "VLLM_ASCEND_GRAPH_EVENT_ORDERING",
    "VLLM_ASCEND_EAGLE_PRESERVE_TARGET_HIDDEN",
    "VLLM_ASCEND_EAGLE_ISOLATE_SHARED_MODULES",
    "VLLM_ASCEND_EAGLE_TARGET_WIDTH",
    "VLLM_ASCEND_EAGLE_TREE_GRAPH_COMMIT",
    "VLLM_ASCEND_EAGLE_ZERO_DRAFT_KV_FIRST_STEP",
    "VLLM_ASCEND_EAGLE_DRAFT_TRACE",
    "VLLM_ASCEND_EAGLE_DRAFT_IO_TRACE_DIR",
    "VLLM_ASCEND_EAGLE_TARGET_HIDDEN_TRACE_DIR",
    "VLLM_ASCEND_EAGLE_TARGET_ARGMAX_TRACE_PATH",
)

DRAFT_ENVIRONMENT_NAMES = (
    "VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_SIZE",
    "VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_IDS_PATH",
    "VLLM_ASCEND_DRAFT_LM_HEAD_W8A16",
    "VLLM_ASCEND_DRAFT_LM_HEAD_W8A8",
    "VLLM_ASCEND_DRAFT_TARGET_ACTIVE_VOCAB_IDS_PATH",
    "VSPEC_DRAFT_BODY_W8A16",
)

SERVE_CONFIG_KEYS = frozenset(
    {
        "target_model",
        "draft_model",
        "model_registry",
        "method",
        "served_model_name",
        "gamma",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_model_len",
        "dtype",
        "block_size",
        "tensor_parallel_size",
        "draft_tensor_parallel_size",
        "gpu_memory_utilization",
        "host",
        "port",
        "device",
        "graph_mode",
        "generation_config",
        "async_scheduling",
        "prefix_caching",
        "chunked_prefill",
        "shared_tokenizer_padding",
        "merged_full",
        "merged_full_max_batch",
        "capture_sizes",
        "capture_policy",
        "disable_log_stats",
        "draft_active_vocab_size",
        "draft_active_vocab_ids",
        "draft_target_active_vocab_ids",
        "draft_lm_head_quantization",
        "draft_body_quantization",
        "draft_parallel_graph_updates",
        "draft_target_parallel_graph_updates",
        "draft_exact_repetition_topk",
        "draft_exact_repetition_trace",
        "draft_exact_repetition_sync_proof",
        "eagle_tree_width",
        "eagle_draft_active_vocab_size",
        "eagle_draft_active_vocab_ids",
        "eagle_target_active_vocab_ids",
        "eagle_draft_lm_head_quantization",
        "eagle_disable_draft_torch_compile",
        "eagle_relaxed_accept_topk",
        "eagle_relaxed_accept_after_tokens",
        "confidence_accept_margin",
        "confidence_accept_from_position",
        "confidence_accept_after_tokens",
        "confidence_protected_token_ids",
        "eagle_spec_metadata_cache",
        "eagle_uniform_state_kernel",
        "graph_event_ordering",
        "eagle_preserve_target_hidden",
        "eagle_isolate_shared_modules",
        "eagle_target_width",
        "eagle_tree_graph_commit",
        "eagle_zero_draft_kv_first_step",
        "eagle_draft_trace",
        "eagle_draft_io_trace_dir",
        "eagle_target_hidden_trace_dir",
        "eagle_target_argmax_trace_path",
        "adaptive_speculation",
        "adaptive_policy",
        "adaptive_profile",
        "adaptive_ewma_weight",
        "adaptive_control_interval",
        "adaptive_hysteresis",
        "adaptive_min_gamma",
        "adaptive_min_observations",
        "adaptive_max_gamma_step",
        "adaptive_latency_calibration",
        "adaptive_latency_ewma_weight",
        "adaptive_gamma0_mode",
        "adaptive_trace",
        "adaptive_full_graph",
        "adaptive_async",
        "adaptive_online_window",
        "adaptive_online_exploration",
        "adaptive_online_warmup_samples",
        "adaptive_online_warmup_return",
        "adaptive_refill_batch",
        "adaptive_entropy_stop",
        "adaptive_entropy_topk",
        "adaptive_entropy_threshold",
        "adaptive_entropy_scale",
        "vllm_source",
        "ascend_source",
        "vllm_executable",
        "extra_args",
    }
)


def positive_int(value: str | int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def nonnegative_int(value: str | int) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value}")
    return parsed


def memory_utilization(value: str | float) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("gpu-memory-utilization must be in (0, 1]")
    return parsed


def unit_float(value: str | float) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("expected a value in (0, 1]")
    return parsed


def nonnegative_float(value: str | float) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative value")
    return parsed


def _power_of_two_buckets(limit: int) -> list[int]:
    buckets: list[int] = []
    bucket = 1
    while bucket < limit:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(limit)
    return buckets


def generate_capture_sizes(
    max_num_seqs: int,
    gamma: int,
    method: str = "draft_model",
    policy: str = "auto",
    target_width: bool = False,
    dynamic_widths: bool = False,
    request_step: int | None = None,
) -> list[int]:
    """Generate request, Draft, and Target verification graph buckets."""
    verification_width = gamma if target_width else gamma + 1
    if policy == "power2":
        return _power_of_two_buckets(max_num_seqs * verification_width)
    if policy == "steady":
        return sorted({1, max_num_seqs, max_num_seqs * verification_width})
    if policy == "exact" or (
        policy == "auto" and (method in {"eagle", "eagle3", "dflash"} or dynamic_widths)
    ):
        if request_step is not None:
            if request_step <= 0:
                raise ValueError("request_step must be positive")
            request_buckets = list(range(request_step, max_num_seqs + 1, request_step))
            request_buckets.extend(size for size in (1, 2, 4, 8) if size <= max_num_seqs)
        else:
            request_buckets = [size for size in EAGLE_REQUEST_CAPTURE_SIZES if size <= max_num_seqs]
        if max_num_seqs not in request_buckets:
            request_buckets.append(max_num_seqs)
    elif policy == "auto":
        request_buckets = _power_of_two_buckets(max_num_seqs)
    else:
        raise ValueError(f"unsupported capture policy: {policy}")
    verification_widths = (
        range(2, verification_width + 1) if dynamic_widths else (verification_width,)
    )
    return sorted(
        set(request_buckets)
        | {batch_size * width for batch_size in request_buckets for width in verification_widths}
    )


def load_config(path: Path | None) -> tuple[dict[str, Any], dict[str, str]]:
    if path is None:
        return {}, {}
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    serve = dict(document.get("serve", {}))
    unknown = sorted(set(serve) - SERVE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown [serve] option(s): {', '.join(unknown)}")
    environment = {str(name): str(value) for name, value in document.get("env", {}).items()}
    return serve, environment


def build_parser(defaults: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-vspec",
        description="Launch vLLM-HUST with the vSpec plugin.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--protocol",
        choices=tuple(SERVE_PROTOCOLS),
        help="Apply a reproducible serving protocol before config and CLI overrides.",
    )
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument(
        "--model-registry",
        type=Path,
        help="Override the install-time default model registry.",
    )
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_ALIASES),
        default="draft",
    )
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--gamma",
        type=positive_int,
        default=4,
        help="Maximum speculative tokens; Adaptive chooses a gamma in 1..4 by default.",
    )
    parser.add_argument("--max-num-seqs", type=positive_int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=8192)
    parser.add_argument("--max-model-len", type=positive_int, default=32768)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--block-size", type=positive_int, default=128)
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=positive_int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=memory_utilization,
        default=0.85,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8000)
    parser.add_argument("--device")
    parser.add_argument(
        "--graph-mode",
        choices=(
            "eager",
            "piecewise",
            "full-decode-only",
            "full-and-piecewise",
            "full",
        ),
        default="full-decode-only",
    )
    parser.add_argument(
        "--generation-config",
        choices=("vllm", "auto"),
        default="vllm",
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shared-tokenizer-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--merged-full",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--merged-full-max-batch",
        type=nonnegative_int,
        default=0,
    )
    parser.add_argument("--capture-sizes", nargs="+", type=positive_int)
    parser.add_argument(
        "--capture-policy",
        choices=("auto", "exact", "power2", "steady"),
        default="auto",
    )
    parser.add_argument(
        "--disable-log-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--draft-active-vocab-size",
        type=nonnegative_int,
        default=0,
    )
    parser.add_argument("--draft-active-vocab-ids", type=Path)
    parser.add_argument("--draft-target-active-vocab-ids", type=Path)
    parser.add_argument(
        "--draft-lm-head-quantization",
        choices=("none", "w8a16", "w8a8"),
        default="none",
    )
    parser.add_argument(
        "--draft-body-quantization",
        choices=("none", "w8a16"),
        default="none",
    )
    parser.add_argument(
        "--draft-parallel-graph-updates",
        type=nonnegative_int,
        default=0,
        help="Number of parallel Draft graph-update workers (0 disables it).",
    )
    parser.add_argument(
        "--draft-target-parallel-graph-updates",
        type=nonnegative_int,
        default=0,
        help="Number of parallel Target graph-update workers (0 disables it).",
    )
    parser.add_argument(
        "--draft-exact-repetition-topk",
        type=nonnegative_int,
        default=0,
        help="Exact repetition-aware greedy Top-K candidate count (0 disables it).",
    )
    parser.add_argument(
        "--draft-exact-repetition-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--draft-exact-repetition-sync-proof",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Synchronize exact Top-K proof before accepting a candidate; "
            "disable only for bounded-candidate performance experiments."
        ),
    )
    parser.add_argument("--eagle-tree-width", type=positive_int, default=1)
    parser.add_argument(
        "--eagle-draft-active-vocab-size",
        type=nonnegative_int,
        default=0,
    )
    parser.add_argument("--eagle-draft-active-vocab-ids", type=Path)
    parser.add_argument("--eagle-target-active-vocab-ids", type=Path)
    parser.add_argument(
        "--eagle-draft-lm-head-quantization",
        choices=("none", "w8a16", "w8a8"),
        default="none",
    )
    parser.add_argument(
        "--eagle-disable-draft-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-relaxed-accept-topk",
        type=positive_int,
        default=1,
    )
    parser.add_argument(
        "--eagle-relaxed-accept-after-tokens",
        type=nonnegative_int,
        default=0,
    )
    parser.add_argument(
        "--confidence-accept-margin",
        type=nonnegative_float,
        help=(
            "Accept a Draft token when its Target logit is within this margin "
            "of Target argmax. This relaxes exact greedy equivalence."
        ),
    )
    parser.add_argument(
        "--confidence-accept-from-position",
        type=nonnegative_int,
        default=0,
    )
    parser.add_argument(
        "--confidence-accept-after-tokens",
        type=nonnegative_int,
        default=0,
    )
    parser.add_argument(
        "--confidence-protected-token-ids",
        nargs="*",
        type=nonnegative_int,
        default=[],
    )
    parser.add_argument(
        "--eagle-spec-metadata-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-uniform-state-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--graph-event-ordering",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-preserve-target-hidden",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-isolate-shared-modules",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-target-width",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-tree-graph-commit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-zero-draft-kv-first-step",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eagle-draft-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eagle-draft-io-trace-dir", type=Path)
    parser.add_argument("--eagle-target-hidden-trace-dir", type=Path)
    parser.add_argument("--eagle-target-argmax-trace-path", type=Path)
    parser.add_argument(
        "--adaptive-speculation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the vSpec Adaptive closed-loop gamma controller (default: enabled).",
    )
    parser.add_argument(
        "--adaptive-policy",
        choices=("profile", "online"),
        default="online",
        help=(
            "profile uses a fitted latency model; online learns directly from "
            "completed serving steps and needs no offline profile."
        ),
    )
    parser.add_argument("--adaptive-profile", type=Path)
    parser.add_argument(
        "--adaptive-ewma-weight",
        type=unit_float,
        default=0.1,
    )
    parser.add_argument(
        "--adaptive-control-interval",
        type=positive_int,
        default=4,
    )
    parser.add_argument(
        "--adaptive-hysteresis",
        type=nonnegative_float,
        default=0.05,
    )
    parser.add_argument(
        "--adaptive-min-gamma",
        type=nonnegative_int,
        default=1,
        help=(
            "Minimum adaptive gamma. Use 0 to enable the target-only mode "
            "selected by --adaptive-gamma0-mode."
        ),
    )
    parser.add_argument(
        "--adaptive-min-observations",
        type=positive_int,
        default=32,
    )
    parser.add_argument(
        "--adaptive-max-gamma-step",
        type=positive_int,
        default=1,
    )
    parser.add_argument(
        "--adaptive-latency-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure completed decode steps for online reward or refine the "
            "offline latency profile."
        ),
    )
    parser.add_argument(
        "--adaptive-latency-ewma-weight",
        type=unit_float,
        default=0.2,
    )
    parser.add_argument(
        "--adaptive-gamma0-mode",
        choices=("sticky", "sync"),
        default="sticky",
        help=(
            "sticky is true target-only; sync runs a one-token shadow Draft "
            "step so the same request can later resume speculation."
        ),
    )
    parser.add_argument(
        "--adaptive-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--adaptive-full-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive-async",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive-online-window",
        type=positive_int,
        default=32,
        help="Sliding reward samples retained per batch bucket and gamma.",
    )
    parser.add_argument(
        "--adaptive-online-exploration",
        type=nonnegative_float,
        default=0.01,
        help="Online exploration strength; use 0 for greedy selection.",
    )
    parser.add_argument(
        "--adaptive-online-warmup-samples",
        type=positive_int,
        default=1,
        help="Stable decode samples collected per gamma before optimization.",
    )
    parser.add_argument(
        "--adaptive-online-warmup-return",
        choices=("best", "incumbent"),
        default="best",
        help=(
            "Gamma selected after the initial sweep: measured best or the "
            "configured maximum incumbent."
        ),
    )
    parser.add_argument(
        "--adaptive-refill-batch",
        type=nonnegative_int,
        default=0,
        help="Waiting-request refill group; 0 selects the method-tuned default.",
    )
    parser.add_argument(
        "--adaptive-entropy-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop each request's current Draft round when confidence drops.",
    )
    parser.add_argument("--adaptive-entropy-topk", type=positive_int, default=2)
    parser.add_argument(
        "--adaptive-entropy-threshold",
        type=nonnegative_float,
        default=0.3,
        help="Current-round uncertainty threshold.",
    )
    parser.add_argument(
        "--adaptive-entropy-scale",
        type=unit_float,
        default=0.15,
        help="Scale applied to Draft token entropy before thresholding.",
    )
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--ascend-source", type=Path)
    parser.add_argument("--vllm-executable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to vllm serve (place after --).",
    )
    parser.set_defaults(**dict(defaults or {}))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, dict[str, str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_parser.add_argument("--protocol", choices=tuple(SERVE_PROTOCOLS))
    config_namespace, _ = config_parser.parse_known_args(raw_args)
    configured_defaults, environment = load_config(config_namespace.config)
    defaults = dict(SERVE_PROTOCOLS.get(config_namespace.protocol, {}))
    defaults.update(configured_defaults)
    parser = build_parser(defaults)
    namespace = parser.parse_args(raw_args)
    namespace.method = METHOD_ALIASES[namespace.method]
    if namespace.protocol == "arc-easy" and not namespace.served_model_name:
        namespace.served_model_name = ARC_EASY_SERVED_MODEL_NAMES.get(namespace.method)
    if not namespace.target_model:
        parser.error("--target-model is required")
    if not namespace.draft_model:
        model_environment = dict(os.environ)
        model_environment.update(environment)
        try:
            default_model = resolve_default_model(
                namespace.method,
                registry_path=namespace.model_registry,
                environment=model_environment,
            )
        except ModelStoreError as exc:
            parser.error(str(exc))
        if default_model is None:
            if namespace.method in {"draft_model", "eagle"}:
                parser.error(
                    f"no installed default model for {namespace.method}; run "
                    "'./manage.sh models' or pass --draft-model"
                )
            parser.error(f"--draft-model is required for --method {namespace.method}")
        namespace.draft_model = str(default_model)
    adaptive_explicitly_enabled = "--adaptive-speculation" in raw_args or (
        "adaptive_speculation" in defaults and bool(defaults["adaptive_speculation"])
    )
    adaptive_policy_explicit = "--adaptive-policy" in raw_args or ("adaptive_policy" in defaults)
    if namespace.adaptive_profile is not None and not adaptive_policy_explicit:
        namespace.adaptive_policy = "profile"
    if (
        namespace.method == "dflash"
        and namespace.adaptive_speculation
        and namespace.adaptive_policy == "online"
    ):
        if adaptive_explicitly_enabled or adaptive_policy_explicit:
            parser.error(
                "--adaptive-policy online currently supports serial Draft, EAGLE, and EAGLE3 only"
            )
        namespace.adaptive_speculation = False
    if namespace.draft_active_vocab_size and namespace.draft_active_vocab_ids is not None:
        parser.error(
            "--draft-active-vocab-size and --draft-active-vocab-ids are mutually exclusive"
        )
    if (
        namespace.eagle_draft_active_vocab_size
        and namespace.eagle_draft_active_vocab_ids is not None
    ):
        parser.error(
            "--eagle-draft-active-vocab-size and "
            "--eagle-draft-active-vocab-ids are mutually exclusive"
        )
    if namespace.eagle_tree_width > 1:
        if namespace.method != "eagle":
            parser.error("--eagle-tree-width only supports --method eagle")
        if namespace.gamma % namespace.eagle_tree_width:
            parser.error("gamma must be divisible by --eagle-tree-width")
        if namespace.eagle_draft_lm_head_quantization != "none":
            parser.error("quantized EAGLE Draft LM head does not support tree mode")
    if namespace.adaptive_speculation:
        if namespace.adaptive_entropy_stop and namespace.adaptive_policy != "online":
            parser.error("--adaptive-entropy-stop requires --adaptive-policy online")
        if namespace.adaptive_policy == "profile":
            if namespace.adaptive_profile is None:
                parser.error("--adaptive-profile is required with --adaptive-policy profile")
            from .adaptive.profile import AdaptiveProfile

            try:
                adaptive_profile = AdaptiveProfile.load(namespace.adaptive_profile)
            except ValueError as exc:
                parser.error(str(exc))
            if namespace.gamma > adaptive_profile.max_speculative_tokens:
                parser.error(
                    "--gamma exceeds adaptive profile max_speculative_tokens "
                    f"({adaptive_profile.max_speculative_tokens})"
                )
            expected_parallel_profile = namespace.method == "dflash"
            if adaptive_profile.draft_parallel != expected_parallel_profile:
                parser.error(
                    "adaptive profile draft_parallel does not match --method: "
                    f"expected {str(expected_parallel_profile).lower()} for "
                    f"{namespace.method}"
                )
        else:
            if namespace.adaptive_profile is not None:
                parser.error("--adaptive-profile cannot be used with --adaptive-policy online")
            if namespace.method == "dflash":
                parser.error(
                    "--adaptive-policy online currently supports serial "
                    "Draft, EAGLE, and EAGLE3 only"
                )
            if not namespace.adaptive_latency_calibration:
                parser.error("--adaptive-policy online requires --adaptive-latency-calibration")
            if namespace.adaptive_min_gamma == 0 and namespace.adaptive_gamma0_mode != "sync":
                parser.error(
                    "online gamma 0 requires --adaptive-gamma0-mode sync so speculation can resume"
                )
        if namespace.adaptive_min_gamma > namespace.gamma:
            parser.error("--adaptive-min-gamma cannot exceed --gamma")
        if namespace.eagle_tree_width > 1:
            parser.error("vSpec Adaptive does not support EAGLE tree mode")
        if namespace.eagle_target_width:
            parser.error("vSpec Adaptive does not support EAGLE Target-width mode")
        if not namespace.adaptive_async:
            namespace.async_scheduling = False
        namespace.merged_full = (
            namespace.adaptive_full_graph
            and namespace.method == "draft_model"
            and namespace.graph_mode
            in {"full", "full-decode-only", "full-and-piecewise"}
        )
        if not namespace.merged_full:
            namespace.merged_full_max_batch = 0
        if namespace.graph_mode != "eager" and not namespace.adaptive_full_graph:
            namespace.graph_mode = "piecewise"
    eagle_only_requested = (
        namespace.eagle_draft_active_vocab_size > 0
        or namespace.eagle_draft_active_vocab_ids is not None
        or namespace.eagle_target_active_vocab_ids is not None
        or namespace.eagle_draft_lm_head_quantization != "none"
        or namespace.eagle_disable_draft_torch_compile
        or namespace.eagle_relaxed_accept_topk > 1
        or namespace.eagle_relaxed_accept_after_tokens > 0
        or namespace.eagle_target_width
        or namespace.eagle_tree_graph_commit
        or namespace.eagle_zero_draft_kv_first_step
        or namespace.eagle_draft_trace
        or namespace.eagle_draft_io_trace_dir is not None
        or namespace.eagle_target_hidden_trace_dir is not None
        or namespace.eagle_target_argmax_trace_path is not None
    )
    if eagle_only_requested and namespace.method != "eagle":
        parser.error("the selected EAGLE optimization only supports --method eagle")
    draft_only_requested = (
        namespace.draft_active_vocab_size > 0
        or namespace.draft_active_vocab_ids is not None
        or namespace.draft_target_active_vocab_ids is not None
        or namespace.draft_lm_head_quantization != "none"
        or namespace.draft_body_quantization != "none"
        or namespace.draft_parallel_graph_updates > 0
        or namespace.draft_target_parallel_graph_updates > 0
        or namespace.draft_exact_repetition_topk > 0
        or namespace.draft_exact_repetition_trace
        or not namespace.draft_exact_repetition_sync_proof
    )
    if draft_only_requested and namespace.method != "draft_model":
        parser.error("the selected Draft optimization only supports --method draft")
    if namespace.method != "draft_model":
        namespace.shared_tokenizer_padding = False
        namespace.merged_full = False
        namespace.merged_full_max_batch = 0
    if namespace.extra_args and namespace.extra_args[0] == "--":
        namespace.extra_args = namespace.extra_args[1:]
    return namespace, environment


def build_vllm_command(options: argparse.Namespace) -> list[str]:
    environment_vllm = Path(sys.executable).with_name("vllm")
    executable = (
        options.vllm_executable
        or shutil.which("vllm")
        or (str(environment_vllm) if environment_vllm.is_file() else None)
    )
    if not executable:
        raise RuntimeError("vllm executable was not found in PATH")

    served_model_name = options.served_model_name or Path(options.target_model).name
    graph_enabled = options.graph_mode != "eager"
    speculative_config = {
        "method": options.method,
        "model": options.draft_model,
        "draft_tensor_parallel_size": options.draft_tensor_parallel_size,
        "num_speculative_tokens": options.gamma,
        "enforce_eager": not graph_enabled,
    }
    if options.method == "draft_model":
        speculative_config["use_heterogeneous_vocab"] = options.shared_tokenizer_padding
    command = [
        executable,
        "serve",
        options.target_model,
        "--host",
        options.host,
        "--port",
        str(options.port),
        "--served-model-name",
        served_model_name,
        "--dtype",
        options.dtype,
        "--kv-cache-dtype",
        "auto",
        "--block-size",
        str(options.block_size),
        "--tensor-parallel-size",
        str(options.tensor_parallel_size),
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--max-model-len",
        str(options.max_model_len),
        "--gpu-memory-utilization",
        str(options.gpu_memory_utilization),
        "--max-num-seqs",
        str(options.max_num_seqs),
        "--max-num-batched-tokens",
        str(options.max_num_batched_tokens),
        "--seed",
        "0",
        "--scheduling-policy",
        "fcfs",
        "--distributed-executor-backend",
        "mp",
        "--disable-custom-all-reduce",
        "--no-trust-remote-code",
        "--load-format",
        "auto",
        "--generation-config",
        options.generation_config,
        "--no-enable-log-requests",
        "--uvicorn-log-level",
        "info",
        "--speculative-config",
        json.dumps(speculative_config, separators=(",", ":")),
    ]
    if options.disable_log_stats:
        command.append("--disable-log-stats")
    command.append(
        "--enable-prefix-caching" if options.prefix_caching else "--no-enable-prefix-caching"
    )
    command.append(
        "--enable-chunked-prefill" if options.chunked_prefill else "--no-enable-chunked-prefill"
    )
    command.append("--async-scheduling" if options.async_scheduling else "--no-async-scheduling")
    command.append("--no-enforce-eager" if graph_enabled else "--enforce-eager")

    if graph_enabled:
        graph_modes = {
            "piecewise": "PIECEWISE",
            "full-decode-only": "FULL_DECODE_ONLY",
            "full-and-piecewise": "FULL_AND_PIECEWISE",
            "full": "FULL",
        }
        graph_mode = graph_modes[options.graph_mode]
        compilation_config = {"mode": 3, "cudagraph_mode": graph_mode}
        capture_sizes = options.capture_sizes or generate_capture_sizes(
            options.max_num_seqs,
            options.gamma,
            options.method,
            options.capture_policy,
            options.eagle_target_width,
            options.adaptive_speculation,
        )
        command.extend(
            [
                "--compilation-config",
                json.dumps(compilation_config, separators=(",", ":")),
                "--cudagraph-capture-sizes",
                *(str(size) for size in sorted(set(capture_sizes))),
            ]
        )
    command.extend(options.extra_args or [])
    return command


def build_environment(
    options: argparse.Namespace,
    configured_environment: Mapping[str, str],
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(configured_environment)
    eagle_draft_active_vocab = (
        options.eagle_draft_active_vocab_size > 0
        or options.eagle_draft_active_vocab_ids is not None
        or options.eagle_draft_lm_head_quantization != "none"
    )
    eagle_target_active_vocab = options.eagle_target_active_vocab_ids is not None
    draft_active_vocab = (
        options.draft_active_vocab_size > 0
        or options.draft_active_vocab_ids is not None
        or options.draft_lm_head_quantization != "none"
    )
    draft_target_active_vocab = options.draft_target_active_vocab_ids is not None
    plugin_settings = PluginSettings(
        enabled=True,
        method=options.method,
        assume_shared_tokenizer=options.shared_tokenizer_padding,
        use_merged_full=options.merged_full,
        merged_full_max_batch=options.merged_full_max_batch,
        max_num_seqs=options.max_num_seqs,
        draft_active_vocab=draft_active_vocab,
        draft_target_active_vocab=draft_target_active_vocab,
        draft_parallel_graph_updates=options.draft_parallel_graph_updates,
        draft_target_parallel_graph_updates=(
            options.draft_target_parallel_graph_updates
        ),
        draft_exact_repetition_topk=options.draft_exact_repetition_topk,
        draft_exact_repetition_trace=options.draft_exact_repetition_trace,
        draft_exact_repetition_sync_proof=(
            options.draft_exact_repetition_sync_proof
        ),
        eagle_tree_width=options.eagle_tree_width,
        eagle_draft_active_vocab=eagle_draft_active_vocab,
        eagle_target_active_vocab=eagle_target_active_vocab,
        eagle_relaxed_accept_topk=options.eagle_relaxed_accept_topk,
        confidence_accept_margin=options.confidence_accept_margin,
        confidence_accept_from_position=options.confidence_accept_from_position,
        confidence_accept_after_tokens=options.confidence_accept_after_tokens,
        confidence_protected_token_ids=tuple(options.confidence_protected_token_ids),
        adaptive_speculation=options.adaptive_speculation,
        adaptive_policy=options.adaptive_policy,
        adaptive_profile_path=(
            str(options.adaptive_profile.resolve()) if options.adaptive_profile is not None else ""
        ),
        adaptive_max_gamma=options.gamma,
        adaptive_min_gamma=options.adaptive_min_gamma,
        adaptive_ewma_weight=options.adaptive_ewma_weight,
        adaptive_control_interval=options.adaptive_control_interval,
        adaptive_hysteresis=options.adaptive_hysteresis,
        adaptive_min_observations=options.adaptive_min_observations,
        adaptive_max_gamma_step=options.adaptive_max_gamma_step,
        adaptive_latency_calibration=(options.adaptive_latency_calibration),
        adaptive_latency_ewma_weight=(options.adaptive_latency_ewma_weight),
        adaptive_gamma0_mode=options.adaptive_gamma0_mode,
        adaptive_trace=options.adaptive_trace,
        adaptive_full_graph=options.adaptive_full_graph,
        adaptive_async=options.adaptive_async,
        adaptive_online_window=options.adaptive_online_window,
        adaptive_online_exploration=options.adaptive_online_exploration,
        adaptive_online_warmup_samples=(options.adaptive_online_warmup_samples),
        adaptive_online_warmup_return=(options.adaptive_online_warmup_return),
        adaptive_refill_batch=options.adaptive_refill_batch,
        adaptive_entropy_stop=options.adaptive_entropy_stop,
        adaptive_entropy_topk=options.adaptive_entropy_topk,
        adaptive_entropy_threshold=options.adaptive_entropy_threshold,
        adaptive_entropy_scale=options.adaptive_entropy_scale,
    )
    environment.update(plugin_settings.as_environment())
    for name in DRAFT_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    for name in EAGLE_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    if options.method == "draft_model":
        if options.draft_body_quantization == "w8a16":
            environment["VSPEC_DRAFT_BODY_W8A16"] = "1"
        if options.draft_active_vocab_size > 0:
            environment["VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_SIZE"] = str(
                options.draft_active_vocab_size
            )
        if options.draft_lm_head_quantization != "none":
            suffix = options.draft_lm_head_quantization.upper()
            environment[f"VLLM_ASCEND_DRAFT_LM_HEAD_{suffix}"] = "1"
        if options.draft_active_vocab_ids is not None:
            environment["VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_IDS_PATH"] = str(
                options.draft_active_vocab_ids
            )
        if options.draft_target_active_vocab_ids is not None:
            environment["VLLM_ASCEND_DRAFT_TARGET_ACTIVE_VOCAB_IDS_PATH"] = str(
                options.draft_target_active_vocab_ids
            )
    if options.method in {"eagle", "eagle3"}:
        enabled_flags = {
            "VLLM_ASCEND_EAGLE_DISABLE_DRAFT_TORCH_COMPILE": (
                options.eagle_disable_draft_torch_compile
            ),
            "VLLM_ASCEND_EAGLE_SPEC_METADATA_CACHE": (options.eagle_spec_metadata_cache),
            "VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL": (options.eagle_uniform_state_kernel),
            "VLLM_ASCEND_GRAPH_EVENT_ORDERING": options.graph_event_ordering,
            "VLLM_ASCEND_EAGLE_PRESERVE_TARGET_HIDDEN": (options.eagle_preserve_target_hidden),
            "VLLM_ASCEND_EAGLE_ISOLATE_SHARED_MODULES": (options.eagle_isolate_shared_modules),
            "VLLM_ASCEND_EAGLE_TARGET_WIDTH": options.eagle_target_width,
            "VLLM_ASCEND_EAGLE_TREE_GRAPH_COMMIT": (options.eagle_tree_graph_commit),
            "VLLM_ASCEND_EAGLE_ZERO_DRAFT_KV_FIRST_STEP": (options.eagle_zero_draft_kv_first_step),
            "VLLM_ASCEND_EAGLE_DRAFT_TRACE": options.eagle_draft_trace,
        }
        environment.update({name: "1" for name, enabled in enabled_flags.items() if enabled})
        if options.eagle_tree_width > 1:
            environment["VLLM_ASCEND_EAGLE_TREE_WIDTH"] = str(options.eagle_tree_width)
        if options.eagle_relaxed_accept_topk > 1:
            environment["VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_TOPK"] = str(
                options.eagle_relaxed_accept_topk
            )
        if options.eagle_relaxed_accept_after_tokens > 0:
            environment["VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_AFTER_TOKENS"] = str(
                options.eagle_relaxed_accept_after_tokens
            )
        if options.eagle_draft_active_vocab_size > 0:
            environment["VLLM_ASCEND_EAGLE_DRAFT_ACTIVE_VOCAB_SIZE"] = str(
                options.eagle_draft_active_vocab_size
            )
        if options.eagle_draft_lm_head_quantization != "none":
            suffix = options.eagle_draft_lm_head_quantization.upper()
            environment[f"VLLM_ASCEND_EAGLE_DRAFT_LM_HEAD_{suffix}"] = "1"
        if options.eagle_draft_active_vocab_ids is not None:
            environment["VLLM_ASCEND_EAGLE_DRAFT_ACTIVE_VOCAB_IDS_PATH"] = str(
                options.eagle_draft_active_vocab_ids
            )
        if options.eagle_target_active_vocab_ids is not None:
            environment["VLLM_ASCEND_EAGLE_TARGET_ACTIVE_VOCAB_IDS_PATH"] = str(
                options.eagle_target_active_vocab_ids
            )
        diagnostic_paths = {
            "VLLM_ASCEND_EAGLE_DRAFT_IO_TRACE_DIR": (options.eagle_draft_io_trace_dir),
            "VLLM_ASCEND_EAGLE_TARGET_HIDDEN_TRACE_DIR": (options.eagle_target_hidden_trace_dir),
            "VLLM_ASCEND_EAGLE_TARGET_ARGMAX_TRACE_PATH": (options.eagle_target_argmax_trace_path),
        }
        environment.update(
            {name: str(path) for name, path in diagnostic_paths.items() if path is not None}
        )
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if options.graph_mode != "eager":
        # The PyTorch 2.10+ AOT artifact loader is not reliable on the tested
        # Ascend stack. Keep the regular compile cache unless explicitly overridden.
        environment.setdefault("VLLM_USE_AOT_COMPILE", "0")
    if options.device is not None:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(options.device)

    source_paths = [
        str(path) for path in (options.vllm_source, options.ascend_source) if path is not None
    ]
    if source_paths:
        existing_pythonpath = environment.get("PYTHONPATH")
        if existing_pythonpath:
            source_paths.append(existing_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(source_paths)

    allowed_plugins = environment.get("VLLM_PLUGINS")
    if allowed_plugins is not None:
        names = [name for name in allowed_plugins.split(",") if name]
        for required_name in (ASCEND_PLATFORM_PLUGIN, PLUGIN_ENTRY_POINT):
            if required_name not in names:
                names.append(required_name)
        environment["VLLM_PLUGINS"] = ",".join(names)
    return environment


def is_plugin_installed() -> bool:
    return any(
        plugin.name == PLUGIN_ENTRY_POINT for plugin in entry_points(group="vllm.general_plugins")
    )


def main(argv: Sequence[str] | None = None) -> None:
    options, configured_environment = parse_args(argv)
    command = build_vllm_command(options)
    environment = build_environment(options, configured_environment)
    if options.dry_run:
        plugin_environment = PluginSettings.from_environment(environment).as_environment()
        plugin_environment.update(
            {name: environment[name] for name in EAGLE_ENVIRONMENT_NAMES if name in environment}
        )
        print(json.dumps(plugin_environment, indent=2, sort_keys=True))
        print(shlex.join(command))
        return
    if not is_plugin_installed():
        raise SystemExit("vspec entry point is not installed; run ./install.sh first")
    os.execvpe(command[0], command, environment)
