"""Ascend runtime patches used by the optimized serial Draft backend."""

from __future__ import annotations

import copy
import inspect
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import vllm_ascend.ops.register_custom_ops  # noqa: F401
import vllm_ascend.spec_decode.llm_base_proposer as ascend_base_proposer
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from ..config import PluginSettings
from ..host_compat import call_with_supported_kwargs
from .draft_vocab import _configure_serial_draft_active_vocab

PATCH_MARKER = "_vllm_hust_vspec_draft_patched"
PROFILE_MARKER = "_vllm_hust_vspec_draft_profile_patched"
CONTINUATION_GRAPH_MARKER = "_vllm_hust_vspec_draft_continuation_graph_patched"
_DRAFT_PROFILE_STATS: dict[str, dict[str, float | int]] = {}


def _record_draft_profile(name: str, elapsed_ms: float) -> None:
    output = os.environ.get("VSPEC_DRAFT_PROFILE_PATH")
    if not output:
        return
    stats = _DRAFT_PROFILE_STATS.setdefault(
        name,
        {"calls": 0, "total_ms": 0.0, "max_ms": 0.0},
    )
    stats["calls"] = int(stats["calls"]) + 1
    stats["total_ms"] = float(stats["total_ms"]) + elapsed_ms
    stats["max_ms"] = max(float(stats["max_ms"]), elapsed_ms)
    if name != "draft_propose" or int(stats["calls"]) % 16:
        return
    document = {
        key: {
            **value,
            "mean_ms": float(value["total_ms"]) / int(value["calls"]),
        }
        for key, value in sorted(_DRAFT_PROFILE_STATS.items())
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _profile_dispatch_call(
    original_dispatch: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    result = original_dispatch(*args, **kwargs)
    runtime_mode, descriptor = result
    num_tokens = kwargs.get("num_tokens", args[0] if args else -1)
    uniform = kwargs.get(
        "uniform_decode",
        args[1] if len(args) > 1 else False,
    )
    _record_draft_profile(
        "dispatch_"
        f"{runtime_mode.name.lower()}_"
        f"{'uniform' if uniform else 'nonuniform'}_"
        f"in{num_tokens}_pad{descriptor.num_tokens}",
        0.0,
    )
    return result


def _install_draft_profiler() -> bool:
    """Install opt-in wall-time probes around host Draft phases."""
    if not os.environ.get("VSPEC_DRAFT_PROFILE_PATH"):
        return False
    if getattr(ACLGraphWrapper, PROFILE_MARKER, False):
        return False

    original_graph_call = ACLGraphWrapper.__call__
    original_target_update = NPUModelRunner._update_full_graph_params_if_needed
    original_target_sample = NPUModelRunner._sample
    original_bookkeeping = NPUModelRunner._bookkeeping_sync
    original_draft_update = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer._update_full_graph_params
    )
    original_draft_set_inputs = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.set_inputs_first_pass
    )
    original_draft_build_metadata = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.build_draft_attn_metadata
    )
    original_draft_attn_update = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm
    )
    original_draft_run_merged = ascend_base_proposer.AscendSpecDecodeBaseProposer._run_merged_draft
    original_draft_sample = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer._sample_draft_from_logits
    )

    def graph_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = get_forward_context()
        entry = (
            self.concrete_aclgraph_entries.get(context.batch_descriptor)
            if context is not None
            else None
        )
        is_replay = entry is not None and entry.aclgraph is not None
        started_at = time.perf_counter()
        result = original_graph_call(self, *args, **kwargs)
        if is_replay:
            role = "draft" if _EXTRA_CTX.is_draft_model else "target"
            _record_draft_profile(
                f"{role}_graph_replay_call",
                (time.perf_counter() - started_at) * 1000.0,
            )
        return result

    def target_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_target_update(self, *args, **kwargs)
        _record_draft_profile(
            "target_graph_update",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    def target_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_target_sample(self, *args, **kwargs)
        _record_draft_profile(
            "target_sample",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    def bookkeeping(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_bookkeeping(self, *args, **kwargs)
        _record_draft_profile(
            "bookkeeping",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    @wraps(original_draft_update)
    def draft_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_draft_update(self, *args, **kwargs)
        _record_draft_profile(
            "draft_graph_update",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    @wraps(original_draft_set_inputs)
    def draft_set_inputs(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_draft_set_inputs(self, *args, **kwargs)
        _record_draft_profile(
            "draft_set_inputs",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    @wraps(original_draft_build_metadata)
    def draft_build_metadata(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_draft_build_metadata(self, *args, **kwargs)
        _record_draft_profile(
            "draft_build_metadata",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    @wraps(original_draft_attn_update)
    def draft_attn_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_draft_attn_update(self, *args, **kwargs)
        _record_draft_profile(
            "draft_attn_metadata_update",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    @wraps(original_draft_run_merged)
    def draft_run_merged(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_draft_run_merged(self, *args, **kwargs)
        _record_draft_profile(
            "draft_run_merged",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    @wraps(original_draft_sample)
    def draft_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_draft_sample(self, *args, **kwargs)
        _record_draft_profile(
            "draft_sample",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    ACLGraphWrapper.__call__ = graph_call
    NPUModelRunner._update_full_graph_params_if_needed = target_update
    NPUModelRunner._sample = target_sample
    NPUModelRunner._bookkeeping_sync = bookkeeping
    (ascend_base_proposer.AscendSpecDecodeBaseProposer._update_full_graph_params) = draft_update
    (ascend_base_proposer.AscendSpecDecodeBaseProposer.set_inputs_first_pass) = draft_set_inputs
    (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.build_draft_attn_metadata
    ) = draft_build_metadata
    (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm
    ) = draft_attn_update
    (ascend_base_proposer.AscendSpecDecodeBaseProposer._run_merged_draft) = draft_run_merged
    (ascend_base_proposer.AscendSpecDecodeBaseProposer._sample_draft_from_logits) = draft_sample
    setattr(ACLGraphWrapper, PROFILE_MARKER, True)
    return True


def _install_inner_logits_processor_alias(model: Any) -> bool:
    """Bridge the current Ascend reduce-sample lookup for causal LMs."""
    inner_model = getattr(model, "model", None)
    logits_processor = getattr(model, "logits_processor", None)
    if inner_model is None or logits_processor is None or hasattr(inner_model, "logits_processor"):
        return False

    # Avoid registering the same nn.Module under a second parent path. The
    # Ascend proposer only needs the processor's _gather_logits method.
    object.__setattr__(inner_model, "logits_processor", logits_processor)
    return True


def _call_host_query_padding(
    host_method: Any,
    runner: Any,
    query_start_loc: Any,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_reqs: int,
    cudagraph_runtime_mode: Any,
    batch_desc_num_reqs: int | None,
    batch_desc_uniform: bool | None,
) -> int:
    kwargs = {
        "cudagraph_runtime_mode": cudagraph_runtime_mode,
        "batch_desc_num_reqs": batch_desc_num_reqs,
    }
    if "batch_desc_uniform" in inspect.signature(host_method).parameters:
        kwargs["batch_desc_uniform"] = batch_desc_uniform
    return host_method(
        runner,
        query_start_loc,
        num_tokens_padded,
        num_reqs_padded,
        num_reqs,
        **kwargs,
    )


def _bind_host_attn_update(
    host_method: Any,
    proposer: Any,
    draft_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> inspect.BoundArguments:
    return inspect.signature(host_method).bind(
        proposer,
        draft_index,
        *args,
        **kwargs,
    )


def _enable_merged_graph_replay(runnable: Any, enabled: bool) -> bool:
    """Mark an existing full-graph wrapper as safe for merged Draft replay."""
    if not enabled or not isinstance(runnable, ACLGraphWrapper):
        return False
    runnable.use_eagle = True
    return True


def _draft_request_buckets(max_num_seqs: int) -> tuple[int, ...]:
    buckets: list[int] = []
    size = 1
    while size < max_num_seqs:
        buckets.append(size)
        size *= 2
    buckets.append(max_num_seqs)
    configured = os.environ.get("VSPEC_DRAFT_REQUEST_BUCKETS", "")
    if configured:
        requested = {
            int(value.strip())
            for value in configured.split(",")
            if value.strip()
        }
        if any(value <= 0 or value > max_num_seqs for value in requested):
            raise ValueError(
                "VSPEC_DRAFT_REQUEST_BUCKETS values must be in "
                f"[1, {max_num_seqs}]"
            )
        buckets.extend(requested)
    return tuple(sorted(set(buckets)))


def _compact_first_pass_capture_buckets(max_num_tokens: int) -> tuple[int, ...]:
    configured = os.environ.get(
        "VSPEC_DRAFT_COMPACT_FIRST_CAPTURE_BUCKETS",
        "52,56",
    )
    buckets = tuple(
        sorted(
            {
                int(value.strip())
                for value in configured.split(",")
                if value.strip()
            }
        )
    )
    if any(bucket <= 0 or bucket >= max_num_tokens for bucket in buckets):
        raise ValueError(
            "VSPEC_DRAFT_COMPACT_FIRST_CAPTURE_BUCKETS must contain positive "
            f"values below {max_num_tokens}"
        )
    return buckets


def _add_draft_continuation_graph_keys(
    dispatcher: Any,
    cudagraph_mode: CUDAGraphMode,
    base_query_len: int,
    *,
    query_lens: tuple[int, ...] | None = None,
) -> int:
    speculative_config = getattr(
        dispatcher.vllm_config,
        "speculative_config",
        None,
    )
    if (
        getattr(speculative_config, "method", None) != "draft_model"
        or cudagraph_mode.decode_mode() != CUDAGraphMode.FULL
        or not cudagraph_mode.separate_routine()
    ):
        return 0

    if query_lens is None:
        query_lens = (base_query_len + 1,)
    query_lens = tuple(sorted(set(query_lens)))
    if not query_lens or any(query_len <= 0 for query_len in query_lens):
        raise ValueError("Draft continuation query lengths must be positive")
    max_num_seqs = dispatcher.vllm_config.scheduler_config.max_num_seqs
    added = 0
    for query_len in query_lens:
        for num_reqs in _draft_request_buckets(max_num_seqs):
            for num_active_loras in dispatcher._get_lora_cases():
                descriptor = BatchDescriptor(
                    num_tokens=num_reqs * query_len,
                    num_reqs=num_reqs,
                    uniform=True,
                    has_lora=num_active_loras > 0,
                    num_active_loras=num_active_loras,
                )
                if descriptor not in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]:
                    dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, descriptor)
                    added += 1
    compact_first_pass_buckets: tuple[int, ...] = ()
    if (
        os.environ.get("VSPEC_DRAFT_COMPACT_FIRST_PASS") == "1"
        and base_query_len == 3
    ):
        max_num_tokens = max_num_seqs * (base_query_len + 1)
        compact_first_pass_buckets = _compact_first_pass_capture_buckets(
            max_num_tokens
        )
        for num_tokens in compact_first_pass_buckets:
            for num_active_loras in dispatcher._get_lora_cases():
                descriptor = BatchDescriptor(
                    num_tokens=num_tokens,
                    num_reqs=max_num_seqs,
                    uniform=False,
                    has_lora=num_active_loras > 0,
                    num_active_loras=num_active_loras,
                )
                if descriptor not in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]:
                    dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, descriptor)
                    added += 1
    dispatcher._vspec_draft_merged_query_lens = query_lens
    dispatcher._vspec_draft_continuation_query_len = query_lens[-1]
    dispatcher._vspec_draft_compact_first_pass_buckets = (
        compact_first_pass_buckets
    )
    return added


def _select_draft_continuation_graph(
    dispatcher: Any,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int,
    valid_modes: Any,
    invalid_modes: Any,
) -> BatchDescriptor | None:
    compact_first_pass = getattr(
        dispatcher,
        "_vspec_compact_first_pass",
        None,
    )
    if compact_first_pass is not None:
        actual_tokens, padded_tokens, num_reqs = compact_first_pass
        if (
            num_tokens in (actual_tokens, padded_tokens)
            and (valid_modes is None or CUDAGraphMode.FULL in valid_modes)
            and (invalid_modes is None or CUDAGraphMode.FULL not in invalid_modes)
        ):
            return BatchDescriptor(
                num_tokens=padded_tokens,
                num_reqs=num_reqs,
                uniform=False,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
            )

    capture_query_len = getattr(
        dispatcher,
        "_vspec_capture_uniform_query_len",
        None,
    )
    active_num_reqs = getattr(
        dispatcher,
        "_vspec_active_num_reqs",
        None,
    )
    if active_num_reqs is None and capture_query_len is None:
        return None

    query_lens = getattr(
        dispatcher,
        "_vspec_draft_merged_query_lens",
        (),
    )
    query_len = capture_query_len or getattr(
        dispatcher,
        "_vspec_active_uniform_query_len",
        None,
    )
    if query_len is None and active_num_reqs > 0 and num_tokens % active_num_reqs == 0:
        runtime_query_len = num_tokens // active_num_reqs
        if runtime_query_len in query_lens:
            query_len = runtime_query_len
            dispatcher._vspec_active_uniform_query_len = query_len
    if (
        not uniform_decode
        or query_len is None
        or query_len not in query_lens
        or num_tokens % query_len
        or (valid_modes is not None and CUDAGraphMode.FULL not in valid_modes)
        or (invalid_modes is not None and CUDAGraphMode.FULL in invalid_modes)
    ):
        return None

    num_reqs = num_tokens // query_len
    candidates = (
        descriptor
        for descriptor in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
        if descriptor.uniform
        and descriptor.num_reqs is not None
        and descriptor.num_reqs >= num_reqs
        and descriptor.num_tokens == descriptor.num_reqs * query_len
        and descriptor.has_lora == has_lora
        and (not has_lora or descriptor.num_active_loras >= num_active_loras)
    )
    return min(
        candidates,
        key=lambda descriptor: descriptor.num_tokens,
        default=None,
    )


def _install_draft_continuation_graph_support() -> bool:
    if getattr(CudagraphDispatcher, CONTINUATION_GRAPH_MARKER, False):
        return False

    original_initialize = CudagraphDispatcher.initialize_cudagraph_keys
    original_dispatch = CudagraphDispatcher.dispatch
    original_warmup_and_capture = GPUModelRunner._warmup_and_capture

    def initialize_cudagraph_keys(
        self: Any,
        cudagraph_mode: CUDAGraphMode,
        uniform_decode_query_len: int = 1,
    ) -> Any:
        result = original_initialize(
            self,
            cudagraph_mode,
            uniform_decode_query_len,
        )
        _add_draft_continuation_graph_keys(
            self,
            cudagraph_mode,
            uniform_decode_query_len,
        )
        return result

    def dispatch(
        self: Any,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes: Any = None,
        invalid_modes: Any = None,
        **kwargs: Any,
    ) -> Any:
        descriptor = _select_draft_continuation_graph(
            self,
            num_tokens,
            uniform_decode,
            has_lora,
            num_active_loras,
            valid_modes,
            invalid_modes,
        )
        if descriptor is not None:
            if os.environ.get("VSPEC_DRAFT_DISPATCH_TRACE") == "1":
                print(
                    "VSPEC_DRAFT_DISPATCH: "
                    f"source=vspec input={num_tokens} "
                    f"uniform={uniform_decode} "
                    f"active_reqs={getattr(self, '_vspec_active_num_reqs', None)} "
                    f"active_qlen={getattr(self, '_vspec_active_uniform_query_len', None)} "
                    f"capture_qlen={getattr(self, '_vspec_capture_uniform_query_len', None)} "
                    f"output={descriptor}",
                    flush=True,
                )
            return CUDAGraphMode.FULL, descriptor
        result = original_dispatch(
            self,
            num_tokens=num_tokens,
            uniform_decode=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            valid_modes=valid_modes,
            invalid_modes=invalid_modes,
            **kwargs,
        )
        if os.environ.get("VSPEC_DRAFT_DISPATCH_TRACE") == "1" and hasattr(
            self, "_vspec_active_num_reqs"
        ):
            print(
                "VSPEC_DRAFT_DISPATCH: "
                f"source=host input={num_tokens} "
                f"uniform={uniform_decode} "
                f"active_reqs={getattr(self, '_vspec_active_num_reqs', None)} "
                f"active_qlen={getattr(self, '_vspec_active_uniform_query_len', None)} "
                f"output={result}",
                flush=True,
            )
        return result

    def warmup_and_capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        descriptor = args[0] if args else kwargs["desc"]
        compact_first_pass_buckets = getattr(
            self.cudagraph_dispatcher,
            "_vspec_draft_compact_first_pass_buckets",
            (),
        )
        if (
            not descriptor.uniform
            and descriptor.num_reqs == self.scheduler_config.max_num_seqs
            and descriptor.num_tokens in compact_first_pass_buckets
        ):
            dispatcher = self.cudagraph_dispatcher
            had_compact = hasattr(dispatcher, "_vspec_compact_first_pass")
            previous_compact = getattr(
                dispatcher,
                "_vspec_compact_first_pass",
                None,
            )
            dispatcher._vspec_compact_first_pass = (
                descriptor.num_tokens,
                descriptor.num_tokens,
                descriptor.num_reqs,
            )
            try:
                return original_warmup_and_capture(self, *args, **kwargs)
            finally:
                if had_compact:
                    dispatcher._vspec_compact_first_pass = previous_compact
                else:
                    del dispatcher._vspec_compact_first_pass

        merged_query_lens = getattr(
            self.cudagraph_dispatcher,
            "_vspec_draft_merged_query_lens",
            (),
        )
        capture_query_len = (
            descriptor.num_tokens // descriptor.num_reqs
            if descriptor.uniform and descriptor.num_reqs
            else None
        )
        if capture_query_len not in merged_query_lens:
            return original_warmup_and_capture(self, *args, **kwargs)

        original_query_len = self.uniform_decode_query_len
        original_dispatch_query_len = self.cudagraph_dispatcher.uniform_decode_query_len
        had_capture_query_len = hasattr(
            self.cudagraph_dispatcher,
            "_vspec_capture_uniform_query_len",
        )
        previous_capture_query_len = getattr(
            self.cudagraph_dispatcher,
            "_vspec_capture_uniform_query_len",
            None,
        )
        self.uniform_decode_query_len = capture_query_len
        self.cudagraph_dispatcher.uniform_decode_query_len = capture_query_len
        self.cudagraph_dispatcher._vspec_capture_uniform_query_len = capture_query_len
        try:
            return original_warmup_and_capture(self, *args, **kwargs)
        finally:
            self.uniform_decode_query_len = original_query_len
            self.cudagraph_dispatcher.uniform_decode_query_len = original_dispatch_query_len
            if had_capture_query_len:
                self.cudagraph_dispatcher._vspec_capture_uniform_query_len = (
                    previous_capture_query_len
                )
            else:
                del self.cudagraph_dispatcher._vspec_capture_uniform_query_len

    CudagraphDispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys
    CudagraphDispatcher.dispatch = dispatch
    GPUModelRunner._warmup_and_capture = warmup_and_capture
    setattr(CudagraphDispatcher, CONTINUATION_GRAPH_MARKER, True)
    return True


def _compact_piecewise_bucket(proposer: Any, num_tokens: int) -> int:
    capture_sizes = proposer.vllm_config.compilation_config.cudagraph_capture_sizes or ()
    return min(
        (size for size in capture_sizes if size >= num_tokens),
        default=num_tokens,
    )


def _compact_draft_metadata(
    proposer: Any,
    metadata_by_layer: dict[str, Any],
    num_tokens: int,
) -> dict[str, Any]:
    compact_by_layer: dict[str, Any] = {}
    compact_by_id: dict[int, Any] = {}
    for layer_name, metadata in metadata_by_layer.items():
        metadata_id = id(metadata)
        compact = compact_by_id.get(metadata_id)
        if compact is None:
            compact = copy.copy(metadata)
            compact.num_actual_tokens = num_tokens
            compact.num_decode_tokens = num_tokens
            compact.num_decodes = num_tokens
            compact.num_prefills = 0
            compact.max_query_len = 1
            compact.actual_seq_lengths_q = list(range(1, num_tokens + 1))
            compact.query_start_loc = proposer.arange[: num_tokens + 1]
            for attribute in (
                "block_tables",
                "seq_lens",
                "seq_lens_cpu",
                "slot_mapping",
            ):
                value = getattr(compact, attribute, None)
                if value is not None:
                    setattr(compact, attribute, value[:num_tokens])
            seq_lens_list = getattr(compact, "seq_lens_list", None)
            if seq_lens_list is not None:
                compact.seq_lens_list = seq_lens_list[:num_tokens]
            compact_by_id[metadata_id] = compact
        compact_by_layer[layer_name] = compact
    return compact_by_layer


def _uniform_descriptor_request_count(
    descriptor: Any,
    num_tokens: int,
    query_width: int,
) -> int | None:
    if (
        descriptor is None
        or not getattr(descriptor, "uniform", False)
        or getattr(descriptor, "num_reqs", None) is None
    ):
        return None
    num_reqs = int(descriptor.num_reqs)
    if (
        num_reqs <= 0
        or int(getattr(descriptor, "num_tokens", 0)) != num_tokens
        or num_tokens != num_reqs * query_width
    ):
        return None
    return num_reqs


def _trace_compact_graph(proposer: Any, message: str) -> None:
    if os.environ.get("VSPEC_DRAFT_COMPACT_TRACE") != "1":
        return
    trace_count = getattr(proposer, "_vspec_compact_trace_count", 0)
    if trace_count >= 80:
        return
    print(f"VSPEC_DRAFT_COMPACT[{trace_count}]: {message}", flush=True)
    proposer._vspec_compact_trace_count = trace_count + 1


class CompactPiecewiseDraftModel:
    """Run serial continuation steps at request-batch granularity."""

    def __init__(
        self,
        proposer: Any,
        model: Any,
        *,
        force_piecewise: bool = False,
    ) -> None:
        self.proposer = proposer
        self.model_instance = model
        self.force_piecewise = force_piecewise
        self.num_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model_instance, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.num_calls += 1
        forward_context = get_forward_context()
        force_piecewise = (
            self.force_piecewise
            and forward_context is not None
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        )
        if self.num_calls == 1 and not force_piecewise:
            return self.model_instance(*args, **kwargs)

        num_tokens = kwargs["input_ids"].shape[0]
        runtime_mode, batch_descriptor = self.proposer.runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens,
            uniform_decode=False,
            has_lora=False,
        )
        if force_piecewise:
            runtime_mode = CUDAGraphMode.PIECEWISE
            num_tokens_padded = _compact_piecewise_bucket(
                self.proposer,
                num_tokens,
            )
            batch_descriptor = BatchDescriptor(
                num_tokens=num_tokens_padded,
                uniform=False,
            )
        elif runtime_mode != CUDAGraphMode.PIECEWISE:
            return self.model_instance(*args, **kwargs)
        else:
            num_tokens_padded = batch_descriptor.num_tokens
        kwargs["input_ids"] = self.proposer.input_ids[:num_tokens_padded]
        kwargs["positions"] = self.proposer._get_positions(num_tokens_padded)
        if kwargs.get("hidden_states") is not None:
            kwargs["hidden_states"] = self.proposer.hidden_states[:num_tokens_padded]
        if kwargs.get("inputs_embeds") is not None:
            kwargs["inputs_embeds"] = self.proposer.inputs_embeds[:num_tokens_padded]

        original_mode = forward_context.cudagraph_runtime_mode
        original_descriptor = forward_context.batch_descriptor
        original_num_tokens = forward_context.num_tokens
        original_padded_num_tokens = forward_context.padded_num_tokens
        original_extra_num_tokens = _EXTRA_CTX.num_tokens
        forward_context.cudagraph_runtime_mode = runtime_mode
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens_padded,
            uniform=False,
        )
        forward_context.num_tokens = num_tokens_padded
        forward_context.padded_num_tokens = num_tokens_padded
        _EXTRA_CTX.num_tokens = num_tokens_padded
        try:
            return self.model_instance(*args, **kwargs)
        finally:
            forward_context.cudagraph_runtime_mode = original_mode
            forward_context.batch_descriptor = original_descriptor
            forward_context.num_tokens = original_num_tokens
            forward_context.padded_num_tokens = original_padded_num_tokens
            _EXTRA_CTX.num_tokens = original_extra_num_tokens


class CompactMergedDraftModel:
    """Capture continuation forwards at request rather than verify width."""

    def __init__(
        self,
        proposer: Any,
        model: Any,
        compact_num_tokens: int,
    ) -> None:
        self.proposer = proposer
        self.model_instance = model
        self.compact_num_tokens = compact_num_tokens
        self.num_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model_instance, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.num_calls += 1
        forward_context = get_forward_context()
        if (
            self.num_calls == 1
            or forward_context is None
            or forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        ):
            return self.model_instance(*args, **kwargs)

        compact_num_tokens = self.compact_num_tokens
        kwargs["input_ids"] = self.proposer.input_ids[:compact_num_tokens]
        kwargs["positions"] = self.proposer._get_positions(compact_num_tokens)
        if kwargs.get("hidden_states") is not None:
            kwargs["hidden_states"] = self.proposer.hidden_states[:compact_num_tokens]
        if kwargs.get("inputs_embeds") is not None:
            kwargs["inputs_embeds"] = self.proposer.inputs_embeds[:compact_num_tokens]

        original_num_tokens = forward_context.num_tokens
        original_padded_num_tokens = forward_context.padded_num_tokens
        original_attn_metadata = forward_context.attn_metadata
        original_extra_num_tokens = _EXTRA_CTX.num_tokens

        compact_metadata = (
            _compact_draft_metadata(
                self.proposer,
                original_attn_metadata,
                compact_num_tokens,
            )
            if isinstance(original_attn_metadata, dict)
            else {}
        )

        # FIA capture tables are keyed by attention query width. Keep compact
        # continuation tasks in the outer merged graph's bucket so one graph
        # update refreshes every serial step before replay. Other graph buckets
        # retain their own lists after this capture call returns.
        aliases: list[tuple[dict[int, Any], int, Any]] = []
        if forward_context.capturing and compact_num_tokens != original_num_tokens:
            from vllm_ascend.compilation.acl_graph import get_draft_graph_params

            graph_params = get_draft_graph_params()
            if graph_params is not None:
                for mapping in (
                    graph_params.events,
                    graph_params.handles,
                    graph_params.attn_params,
                    graph_params.workspaces,
                ):
                    previous = mapping.get(compact_num_tokens)
                    aliases.append((mapping, compact_num_tokens, previous))
                    mapping[compact_num_tokens] = mapping.get(original_num_tokens)

        forward_context.num_tokens = compact_num_tokens
        forward_context.padded_num_tokens = compact_num_tokens
        if compact_metadata:
            forward_context.attn_metadata = compact_metadata
        _EXTRA_CTX.num_tokens = compact_num_tokens
        try:
            return self.model_instance(*args, **kwargs)
        finally:
            _EXTRA_CTX.num_tokens = original_extra_num_tokens
            forward_context.attn_metadata = original_attn_metadata
            forward_context.padded_num_tokens = original_padded_num_tokens
            forward_context.num_tokens = original_num_tokens
            for mapping, key, previous in aliases:
                mapping[key] = previous


def apply_draft_patches(settings: PluginSettings) -> bool:
    """Install process-local monkey patches once and return whether applied."""
    applied = _install_draft_continuation_graph_support()
    applied = _install_draft_profiler() or applied
    if os.environ.get("VLLM_ASCEND_GRAPH_EVENT_ORDERING") == "1":
        from .eagle_graph import apply_graph_event_ordering_patch

        applied = apply_graph_event_ordering_patch() or applied
    parallel_update_workers = settings.draft_parallel_graph_updates
    if parallel_update_workers == 0:
        parallel_update_workers = int(os.environ.get("VSPEC_DRAFT_PARALLEL_GRAPH_UPDATES", "0"))
    target_parallel_update_workers = settings.draft_target_parallel_graph_updates
    if target_parallel_update_workers == 0:
        target_parallel_update_workers = int(
            os.environ.get("VSPEC_DRAFT_TARGET_PARALLEL_GRAPH_UPDATES", "0")
        )
    if parallel_update_workers > 1 or target_parallel_update_workers > 1:
        from .draft_parallel_update import (
            apply_draft_parallel_graph_update_patch,
        )

        applied = (
            apply_draft_parallel_graph_update_patch(
                parallel_update_workers,
                target_parallel_update_workers,
            )
            or applied
        )
    if os.environ.get("VSPEC_DRAFT_BODY_W8A16") == "1":
        from .draft_body_quant import apply_draft_body_quantization_patch

        applied = apply_draft_body_quantization_patch() or applied
    if settings.draft_target_active_vocab:
        from .eagle_target import apply_target_active_vocab_patch

        applied = apply_target_active_vocab_patch(
            active_ids_environment=("VLLM_ASCEND_DRAFT_TARGET_ACTIVE_VOCAB_IDS_PATH"),
            required_method="draft_model",
            feature_name="Draft Target",
        )
    if getattr(AscendDraftModelProposer, PATCH_MARKER, False):
        return applied

    original_ascend_init = AscendDraftModelProposer.__init__
    original_draft_dummy_run = ascend_base_proposer.AscendSpecDecodeBaseProposer.dummy_run
    original_pad_query_start_loc = NPUModelRunner._pad_query_start_loc_for_fia
    original_run_merged_draft = ascend_base_proposer.AscendSpecDecodeBaseProposer._run_merged_draft
    original_attn_update = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm
    )
    original_draft_graph_update = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer._update_full_graph_params_if_needed
    )
    original_draft_full_graph_update = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer._update_full_graph_params
    )
    original_draft_propose = ascend_base_proposer.AscendSpecDecodeBaseProposer._propose
    original_set_inputs_first_pass = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.set_inputs_first_pass
    )
    original_sample_draft = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer._sample_draft_from_logits
    )
    original_compute_slot_mapping = ascend_base_proposer.compute_new_slot_mapping

    from .eagle_rejection import apply_linear_rejection_patch

    apply_linear_rejection_patch()

    def init_draft_proposer(
        self: Any,
        vllm_config: Any,
        device: Any,
        runner: Any = None,
    ) -> None:
        if settings.assume_shared_tokenizer:
            DraftModelProposer.__init__(self, vllm_config, device, runner)
            # The known Qwen2.5 pair shares token IDs and differs only in LM-head
            # padding, so runtime remapping is unnecessary.
            self.use_heterogeneous_vocab = False
            self.vocab_mapping = None
        else:
            original_ascend_init(self, vllm_config, device, runner)

        # The proposer is constructed before runner._use_aclgraph() is finalized.
        self.use_cuda_graph = (
            not vllm_config.model_config.enforce_eager and not self.speculative_config.enforce_eager
        )
        fused_proposal_repetition = os.environ.get("VSPEC_DRAFT_FUSED_PROPOSAL_REPETITION") == "1"
        self._vspec_fused_proposal_repetition = fused_proposal_repetition
        sparse_proposal_repetition = (
            os.environ.get("VSPEC_DRAFT_SPARSE_PROPOSAL_REPETITION") == "1"
        )
        self._vspec_sparse_proposal_repetition = sparse_proposal_repetition
        if os.environ.get("VSPEC_DRAFT_ALIGN_REPETITION") == "1":
            from .draft_repetition import DraftRepetitionState

            history_width = int(os.environ.get("VSPEC_DRAFT_REPETITION_HISTORY_WIDTH", "512"))
            if history_width <= 0:
                raise ValueError("VSPEC_DRAFT_REPETITION_HISTORY_WIDTH must be positive")
            capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
            capture_limit = vllm_config.compilation_config.max_cudagraph_capture_size or 0
            max_repetition_rows = max(
                self.max_batch_size,
                capture_limit,
                max(capture_sizes, default=0),
            )
            self._vspec_repetition_state = DraftRepetitionState(
                max_batch_size=max_repetition_rows,
                history_width=history_width,
                vocab_size=self.draft_model_config.get_vocab_size(),
                device=self.device,
            )
            self._vspec_repetition_prefix = []
        elif fused_proposal_repetition or sparse_proposal_repetition:
            self._vspec_repetition_prefix = []
        if os.environ.get("VSPEC_DRAFT_HYBRID_SECOND_TOKEN") == "1":
            from .draft_hybrid import DraftHybridSecondTokenState

            history_width = int(
                os.environ.get("VSPEC_DRAFT_HYBRID_HISTORY_WIDTH", "1024")
            )
            capture_sizes = (
                vllm_config.compilation_config.cudagraph_capture_sizes or ()
            )
            capture_limit = (
                vllm_config.compilation_config.max_cudagraph_capture_size or 0
            )
            max_hybrid_rows = max(
                self.max_batch_size,
                capture_limit,
                max(capture_sizes, default=0),
            )
            self._vspec_hybrid_second_token = DraftHybridSecondTokenState(
                max_batch_size=max_hybrid_rows,
                history_width=history_width,
                vocab_size=vllm_config.model_config.get_vocab_size(),
                device=self.device,
            )
        self._compact_merged_continuations = (
            os.environ.get("VSPEC_DRAFT_COMPACT_MERGED_CONTINUATIONS") == "1"
        )
        self._vspec_gamma2_compact_second = (
            os.environ.get("VSPEC_DRAFT_GAMMA2_COMPACT_SECOND") == "1"
        )
        self._vspec_gamma2_unified_compact_second = (
            os.environ.get("VSPEC_DRAFT_GAMMA2_UNIFIED_COMPACT_SECOND") == "1"
        )
        self._vspec_gamma2_precompact_metadata = (
            os.environ.get("VSPEC_DRAFT_GAMMA2_PRECOMPACT_METADATA") == "1"
        )
        self._vspec_gamma3_unified_compact = (
            os.environ.get("VSPEC_DRAFT_GAMMA3_UNIFIED_COMPACT") == "1"
        )
        self._vspec_gamma2_fast_metadata = (
            os.environ.get("VSPEC_DRAFT_GAMMA2_FAST_METADATA") == "1"
            and self.num_speculative_tokens == 2
        )
        if self._vspec_gamma2_fast_metadata:
            metadata_lens = self.seq_lens_group[0].numel()
            pin_memory = bool(self.runner.pin_memory)
            seq_lens_cpu = self.runner.optimistic_seq_lens_cpu
            computed_tokens_cpu = (
                self.runner.input_batch.num_computed_tokens_cpu_tensor
            )
            self._vspec_fast_seq_lens_cpu = torch.empty(
                metadata_lens,
                dtype=seq_lens_cpu.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._vspec_fast_internal_seq_lens_cpu = torch.empty_like(
                self._vspec_fast_seq_lens_cpu,
                pin_memory=pin_memory,
            )
            self._vspec_fast_computed_tokens_cpu = torch.empty(
                metadata_lens,
                dtype=computed_tokens_cpu.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._vspec_fast_positions = torch.empty_like(self.positions)
            self._vspec_fast_query_start_loc_cpu = torch.arange(
                metadata_lens,
                dtype=torch.int32,
                device="cpu",
            )
        self._vspec_compact_first_pass = (
            os.environ.get("VSPEC_DRAFT_COMPACT_FIRST_PASS") == "1"
        )
        if self._vspec_compact_first_pass:
            max_tokens = self.input_ids.shape[0]
            pin_memory = bool(self.runner.pin_memory)
            self._vspec_compact_indices_cpu = torch.empty(
                max_tokens,
                dtype=torch.int64,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._vspec_compact_indices = torch.empty(
                max_tokens,
                dtype=torch.int64,
                device=self.device,
            )
            self._vspec_compact_input_ids = torch.empty_like(self.input_ids)
            self._vspec_compact_positions = torch.empty_like(self.positions)
            self._vspec_compact_slot_mapping = torch.empty_like(
                self.slot_mapping_group[0]
            )
            self._vspec_compact_sample_indices_cpu = torch.empty(
                self.max_batch_size,
                dtype=torch.int32,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._vspec_compact_sample_indices = torch.empty(
                self.max_batch_size,
                dtype=torch.int32,
                device=self.device,
            )
        self._vspec_gamma2_compact_graph_keys: dict[int, int] = {}
        self._vspec_gamma2_compact_num_tokens: dict[int, int] = {}
        self._vspec_gamma2_unified_compact_num_tokens: dict[int, int] = {}
        self._vspec_compact_trace_count = 0
    def draft_dummy_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        repetition_prefix = getattr(self, "_vspec_repetition_prefix", None)
        if repetition_prefix is not None:
            repetition_prefix.clear()
        _install_inner_logits_processor_alias(self.model)
        if settings.draft_active_vocab:
            _configure_serial_draft_active_vocab(self)
        repetition_state = getattr(self, "_vspec_repetition_state", None)
        if repetition_state is not None:
            repetition_state.configure_active_ids(
                getattr(self, "_eagle_draft_active_vocab_ids", None)
            )
        merged_graph_requested = (
            self.use_eagle or settings.use_merged_full or settings.merged_full_max_batch > 0
        )
        # Ascend normally installs this wrapper during load_model(), before the
        # plugin's dummy_run hook. Update it instead of retaining its replay barrier.
        reused_wrapper = _enable_merged_graph_replay(
            self._runnable,
            self.use_cuda_graph and merged_graph_requested,
        )
        if self.use_cuda_graph and not reused_wrapper:
            self.update_stream = torch.npu.Stream()
            self._runnable = call_with_supported_kwargs(
                ACLGraphWrapper,
                self._run_merged_draft,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=merged_graph_requested,
                enable_enpu=self.enable_enpu,
                is_draft_model=True,
            )

        descriptor = kwargs.get("batch_descriptor")
        num_reqs = kwargs.get("num_reqs")
        merged_query_lens = getattr(
            self.runner.cudagraph_dispatcher,
            "_vspec_draft_merged_query_lens",
            (),
        )
        capture_query_len = (
            descriptor.num_tokens // descriptor.num_reqs
            if descriptor is not None and descriptor.uniform and descriptor.num_reqs
            else None
        )
        compact_capture = (
            descriptor is not None
            and not descriptor.uniform
            and descriptor.num_reqs == settings.max_num_seqs
            and descriptor.num_tokens
            in getattr(
                self.runner.cudagraph_dispatcher,
                "_vspec_draft_compact_first_pass_buckets",
                (),
            )
        )
        if (
            num_reqs is None
            or (
                capture_query_len not in merged_query_lens
                and not compact_capture
            )
            or not isinstance(self._runnable, ACLGraphWrapper)
        ):
            return original_draft_dummy_run(self, *args, **kwargs)

        graph_runnable = self._runnable

        def run_with_descriptor_batch(
            *run_args: Any,
            **run_kwargs: Any,
        ) -> Any:
            run_kwargs["batch_size"] = num_reqs
            token_indices = run_kwargs.get("token_indices_to_sample")
            if token_indices is not None:
                token_count = num_reqs * self.extra_slots_per_request
                run_kwargs["token_indices_to_sample"] = token_indices[:token_count]
            return graph_runnable(*run_args, **run_kwargs)

        self._runnable = run_with_descriptor_batch
        try:
            return original_draft_dummy_run(self, *args, **kwargs)
        finally:
            self._runnable = graph_runnable

    def pad_query_start_loc_for_fia(
        self: Any,
        query_start_loc: Any,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: Any = None,
        batch_desc_num_reqs: int | None = None,
        batch_desc_uniform: bool | None = None,
    ) -> int:
        if (
            getattr(self, "_draft_merged_full_padding", False)
            and cudagraph_runtime_mode == CUDAGraphMode.FULL
            and batch_desc_num_reqs is not None
        ):
            if num_reqs > batch_desc_num_reqs:
                raise RuntimeError(
                    "draft graph descriptor has fewer request rows than the "
                    f"runtime batch: actual={num_reqs}, padded={batch_desc_num_reqs}"
                )

            last_loc = int(query_start_loc.np[num_reqs])
            remaining_tokens = num_tokens_padded - last_loc
            remaining_reqs = batch_desc_num_reqs - num_reqs
            if remaining_tokens < 0:
                raise RuntimeError(
                    "draft graph has fewer token rows than the runtime batch: "
                    f"actual={last_loc}, padded={num_tokens_padded}"
                )

            if remaining_reqs:
                base, extra = divmod(remaining_tokens, remaining_reqs)
                cursor = last_loc
                for offset in range(remaining_reqs):
                    cursor += base + (offset < extra)
                    query_start_loc.np[num_reqs + offset + 1] = cursor
            elif remaining_tokens:
                query_start_loc.np[num_reqs] = num_tokens_padded

            query_start_loc.copy_to_gpu()
            return batch_desc_num_reqs

        return _call_host_query_padding(
            original_pad_query_start_loc,
            self,
            query_start_loc,
            num_tokens_padded,
            num_reqs_padded,
            num_reqs,
            cudagraph_runtime_mode,
            batch_desc_num_reqs,
            batch_desc_uniform,
        )

    def compute_draft_slot_mapping(*, cad: Any, **kwargs: Any) -> Any:
        num_query_rows = len(cad.query_start_loc) - 1
        if len(cad.block_table_tensor) != num_query_rows:
            cad = cad.replace(block_table_tensor=cad.block_table_tensor[:num_query_rows])
        return original_compute_slot_mapping(cad=cad, **kwargs)

    def compact_first_pass_inputs(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_set_inputs_first_pass(self, *args, **kwargs)
        if not getattr(self, "_vspec_compact_first_pass", False):
            return result

        bound = inspect.signature(original_set_inputs_first_pass).bind_partial(
            self,
            *args,
            **kwargs,
        )
        rejected = bound.arguments.get("num_rejected_tokens_gpu")
        if rejected is None or bound.arguments.get("num_prefill_reqs", 0):
            return result

        num_tokens, _, metadata, long_seq_args = result
        batch_size = metadata.batch_size()
        if (
            self.num_speculative_tokens != 2
            or batch_size != settings.max_num_seqs
            or num_tokens != batch_size * (self.num_speculative_tokens + 2)
        ):
            return result

        event = getattr(self.runner, "valid_sampled_token_count_event", None)
        counts_cpu = getattr(self.runner, "valid_sampled_token_count_cpu", None)
        if event is None or counts_cpu is None:
            return result
        attempts = getattr(self, "_vspec_compact_first_attempts", 0) + 1
        self._vspec_compact_first_attempts = attempts
        if not event.query():
            if (
                os.environ.get("VSPEC_DRAFT_COMPACT_FIRST_TRACE") == "1"
                and attempts % 100 == 0
            ):
                hits = getattr(self, "_vspec_compact_first_hits", 0)
                print(
                    "VSPEC_DRAFT_COMPACT_FIRST: "
                    f"attempts={attempts} hits={hits} event_pending=1",
                    flush=True,
                )
            return result
        counts = counts_cpu[:batch_size].numpy()
        if ((counts < 0) | (counts > self.num_speculative_tokens + 1)).any():
            return result

        kept_per_request = counts + 1
        compact_tokens = int(kept_per_request.sum())
        capture_buckets = getattr(
            self.runner.cudagraph_dispatcher,
            "_vspec_draft_compact_first_pass_buckets",
            (),
        )
        padded_tokens = next(
            (bucket for bucket in capture_buckets if bucket >= compact_tokens),
            None,
        )
        if padded_tokens is None or padded_tokens >= num_tokens:
            return result
        self._vspec_compact_first_hits = (
            getattr(self, "_vspec_compact_first_hits", 0) + 1
        )
        if (
            os.environ.get("VSPEC_DRAFT_COMPACT_FIRST_TRACE") == "1"
            and attempts % 100 == 0
        ):
            print(
                "VSPEC_DRAFT_COMPACT_FIRST: "
                f"attempts={attempts} "
                f"hits={self._vspec_compact_first_hits} "
                f"tokens={compact_tokens}->{padded_tokens}",
                flush=True,
            )

        indices_np = self._vspec_compact_indices_cpu.numpy()
        query_start_loc_np = metadata.query_start_loc_cpu.numpy()
        cursor = 0
        for req_idx, kept in enumerate(kept_per_request):
            kept = int(kept)
            source_start = req_idx * (self.num_speculative_tokens + 2)
            indices_np[cursor : cursor + kept] = range(
                source_start,
                source_start + kept,
            )
            cursor += kept
            query_start_loc_np[req_idx + 1] = cursor

        compact_indices = self._vspec_compact_indices[:compact_tokens]
        compact_indices.copy_(
            self._vspec_compact_indices_cpu[:compact_tokens],
            non_blocking=True,
        )
        torch.index_select(
            self.input_ids,
            0,
            compact_indices,
            out=self._vspec_compact_input_ids[:compact_tokens],
        )
        torch.index_select(
            self.positions,
            0,
            compact_indices,
            out=self._vspec_compact_positions[:compact_tokens],
        )
        torch.index_select(
            metadata.slot_mapping,
            0,
            compact_indices,
            out=self._vspec_compact_slot_mapping[:compact_tokens],
        )
        self.input_ids[:compact_tokens].copy_(
            self._vspec_compact_input_ids[:compact_tokens]
        )
        self.positions[:compact_tokens].copy_(
            self._vspec_compact_positions[:compact_tokens]
        )
        metadata.query_start_loc.copy_(
            metadata.query_start_loc_cpu,
            non_blocking=True,
        )

        sample_indices_np = self._vspec_compact_sample_indices_cpu.numpy()
        sample_indices_np[:batch_size] = query_start_loc_np[1 : batch_size + 1] - 1
        sample_indices = self._vspec_compact_sample_indices[:batch_size]
        sample_indices.copy_(
            self._vspec_compact_sample_indices_cpu[:batch_size],
            non_blocking=True,
        )

        compact_slot_mapping = self._vspec_compact_slot_mapping[:compact_tokens]
        compact_metadata = metadata.replace(
            query_start_loc=metadata.query_start_loc[: batch_size + 1],
            query_start_loc_cpu=metadata.query_start_loc_cpu[: batch_size + 1],
            num_actual_tokens=compact_tokens,
            num_input_tokens=compact_tokens,
            max_query_len=int(kept_per_request.max()),
            slot_mapping=compact_slot_mapping,
            positions=self.positions[:compact_tokens],
        )
        self.runner.cudagraph_dispatcher._vspec_compact_first_pass = (
            compact_tokens,
            padded_tokens,
            batch_size,
        )
        return compact_tokens, sample_indices, compact_metadata, long_seq_args

    def run_serial_compact_without_hidden_copies(
        self: Any,
        num_input_tokens: int,
        batch_size: int,
        token_indices_to_sample: torch.Tensor,
        target_positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        multi_steps_attn_metadata: list[Any],
        num_tokens: int,
        is_prefill: bool | None = None,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        if sampling_metadata is None:
            sampling_metadata = self.runner.input_batch.sampling_metadata
        if not sampling_metadata.all_greedy:
            return original_run_merged_draft(
                self,
                num_input_tokens,
                batch_size,
                token_indices_to_sample,
                target_positions,
                inputs_embeds,
                multi_steps_attn_metadata,
                num_tokens,
                is_prefill,
                sampling_metadata,
            )
        self._last_draft_probs = None

        model_input_ids = self.input_ids[:num_input_tokens]
        model_positions = self._get_positions(num_input_tokens)
        hybrid_state = getattr(self, "_vspec_hybrid_second_token", None)
        base_token_ids = None
        if hybrid_state is not None:
            base_token_ids = model_input_ids[token_indices_to_sample][
                :batch_size
            ].clone()
        ret_hidden_states = self.model(
            input_ids=model_input_ids,
            positions=model_positions,
            inputs_embeds=inputs_embeds,
        )
        if self.model_returns_tuple():
            last_hidden_states = ret_hidden_states[0]
        else:
            last_hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        logits = self.model.compute_logits(sample_hidden_states)
        logits = self.model.model.logits_processor._gather_logits(logits)
        first_token_ids, first_probs = self._sample_draft_from_logits(
            logits,
            sampling_metadata,
        )
        first_token_ids = first_token_ids.clone()
        if first_probs is not None:
            first_probs = first_probs.clone()

        if hybrid_state is not None:
            assert base_token_ids is not None
            second_token_ids = hybrid_state.propose(
                base_token_ids,
                first_token_ids,
            )
            return torch.stack((first_token_ids, second_token_ids), dim=1)

        forward_context = get_forward_context()
        if forward_context is not None:
            forward_context.moe_layer_index = 0
        split_compact_second = getattr(
            self,
            "_vspec_gamma2_compact_second",
            False,
        )
        unified_compact = (
            getattr(self, "_vspec_gamma2_unified_compact_second", False)
            if self.num_speculative_tokens == 2
            else getattr(self, "_vspec_gamma3_unified_compact", False)
        )
        compact_continuations = split_compact_second or unified_compact
        continuation_num_tokens = num_input_tokens
        if compact_continuations:
            # Continuation steps consume one token per request. Prefill
            # descriptors do not carry ``num_reqs``, so use the proposer batch
            # size there instead of treating every prompt token as a request.
            continuation_num_tokens = batch_size
            if (
                forward_context is not None
                and forward_context.batch_descriptor is not None
                and forward_context.batch_descriptor.num_reqs is not None
            ):
                continuation_num_tokens = forward_context.batch_descriptor.num_reqs
        original_context_num_tokens = (
            forward_context.num_tokens if forward_context is not None else None
        )
        original_context_padded_tokens = (
            forward_context.padded_num_tokens
            if forward_context is not None
            else None
        )
        original_context_descriptor = (
            forward_context.batch_descriptor
            if forward_context is not None
            else None
        )
        original_extra_num_tokens = _EXTRA_CTX.num_tokens
        _EXTRA_CTX.num_accept_tokens = batch_size
        positions = self.positions[token_indices_to_sample]
        draft_token_ids = [first_token_ids]
        draft_probs = [first_probs] if first_probs is not None else None

        for continuation_index in range(1, self.num_speculative_tokens):
            positions = positions + 1
            clamped_positions = torch.where(
                positions >= self.vllm_config.model_config.max_model_len,
                0,
                positions,
            )
            self.input_ids[:batch_size] = draft_token_ids[-1]
            self._set_positions(batch_size, clamped_positions)
            if continuation_index < len(multi_steps_attn_metadata):
                if compact_continuations:
                    multi_steps_attn_metadata[continuation_index] = (
                        _compact_draft_metadata(
                            self,
                            multi_steps_attn_metadata[continuation_index],
                            continuation_num_tokens,
                        )
                    )
                if forward_context is not None:
                    forward_context.attn_metadata = multi_steps_attn_metadata[
                        continuation_index
                    ]

            if forward_context is not None:
                forward_context.moe_layer_index = 0
            if forward_context is not None and compact_continuations:
                forward_context.num_tokens = continuation_num_tokens
                forward_context.padded_num_tokens = continuation_num_tokens
                forward_context.batch_descriptor = BatchDescriptor(
                    num_tokens=continuation_num_tokens,
                    num_reqs=continuation_num_tokens,
                    uniform=True,
                )
                _trace_compact_graph(
                    self,
                    f"continuation-{continuation_index}-pre "
                    f"outer={num_input_tokens} compact={continuation_num_tokens} "
                    f"capturing={forward_context.capturing}",
                )
            _EXTRA_CTX.num_tokens = continuation_num_tokens

            graph_param_aliases: list[tuple[dict[int, Any], int, Any]] = []
            graph_workspace_alias: tuple[dict[int, Any], int, Any, int] | None = None
            missing_alias = object()
            if (
                compact_continuations
                and forward_context is not None
                and forward_context.capturing
                and continuation_num_tokens != num_input_tokens
            ):
                from vllm_ascend.compilation.acl_graph import (
                    get_draft_graph_params,
                )

                graph_params = get_draft_graph_params()
                compact_graph_key = (
                    num_input_tokens
                    if unified_compact
                    else -(num_input_tokens + 1)
                )
                if unified_compact:
                    self._vspec_gamma2_unified_compact_num_tokens[
                        num_input_tokens
                    ] = continuation_num_tokens
                else:
                    self._vspec_gamma2_compact_graph_keys[num_input_tokens] = (
                        compact_graph_key
                    )
                self._vspec_gamma2_compact_num_tokens[num_input_tokens] = (
                    continuation_num_tokens
                )
                for mapping in (
                    graph_params.events,
                    graph_params.handles,
                    graph_params.attn_params,
                ):
                    previous = mapping.get(
                        continuation_num_tokens,
                        missing_alias,
                    )
                    graph_param_aliases.append(
                        (mapping, continuation_num_tokens, previous)
                    )
                    mapping[continuation_num_tokens] = mapping.setdefault(
                        compact_graph_key,
                        [],
                    )
                workspace_mapping = graph_params.workspaces
                previous_workspace = workspace_mapping.get(
                    continuation_num_tokens,
                    missing_alias,
                )
                workspace_mapping.setdefault(compact_graph_key, None)
                workspace_mapping[continuation_num_tokens] = workspace_mapping[
                    compact_graph_key
                ]
                graph_workspace_alias = (
                    workspace_mapping,
                    continuation_num_tokens,
                    previous_workspace,
                    compact_graph_key,
                )

            try:
                ret_hidden_states = self.model(
                    input_ids=self.input_ids[:continuation_num_tokens],
                    positions=self._get_positions(continuation_num_tokens),
                    inputs_embeds=None,
                )
            finally:
                if forward_context is not None and compact_continuations:
                    forward_context.num_tokens = original_context_num_tokens
                    forward_context.padded_num_tokens = original_context_padded_tokens
                    forward_context.batch_descriptor = original_context_descriptor
                _EXTRA_CTX.num_tokens = original_extra_num_tokens
                if graph_workspace_alias is not None:
                    mapping, key, previous, compact_graph_key = graph_workspace_alias
                    if compact_graph_key != num_input_tokens:
                        mapping[compact_graph_key] = mapping.get(key)
                    if previous is missing_alias:
                        mapping.pop(key, None)
                    else:
                        mapping[key] = previous
                for mapping, key, previous in graph_param_aliases:
                    if previous is missing_alias:
                        mapping.pop(key, None)
                    else:
                        mapping[key] = previous

            if self.model_returns_tuple():
                last_hidden_states = ret_hidden_states[0]
            else:
                last_hidden_states = ret_hidden_states
            sample_hidden_states = last_hidden_states[self.arange[:batch_size]]
            logits = self.model.compute_logits(sample_hidden_states)
            logits = self.model.model.logits_processor._gather_logits(logits)
            next_token_ids, next_probs = self._sample_draft_from_logits(
                logits,
                sampling_metadata,
            )
            if continuation_index + 1 < self.num_speculative_tokens:
                next_token_ids = next_token_ids.clone()
            draft_token_ids.append(next_token_ids)
            if draft_probs is not None:
                if next_probs is None:
                    draft_probs = None
                else:
                    draft_probs.append(next_probs.clone())

        if draft_probs is not None:
            self._last_draft_probs = torch.stack(
                draft_probs,
                dim=1,
            ).contiguous()
        return torch.stack(draft_token_ids, dim=1)

    def run_compact_piecewise_draft(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        repetition_prefix = getattr(self, "_vspec_repetition_prefix", None)
        if repetition_prefix is not None:
            repetition_prefix.clear()
        gamma2_fast_path = (
            os.environ.get("VSPEC_DRAFT_GAMMA2_FAST_PATH") == "1"
            and self.num_speculative_tokens == 2
        )
        gamma3_fast_path = (
            os.environ.get("VSPEC_DRAFT_GAMMA3_UNIFIED_COMPACT") == "1"
            and self.num_speculative_tokens == 3
        )
        if (
            (gamma2_fast_path or gamma3_fast_path)
            and self.method == "draft_model"
            and not self.pass_hidden_states_to_model
            and not self.supports_mm_inputs
            and not self.uses_mrope
            and self.use_cuda_graph
            and getattr(self.runner, "dcp_manager", None) is None
            and not ascend_base_proposer.lmhead_tp_enable()
            and ascend_base_proposer.get_ascend_config().enable_reduce_sample
        ):
            return run_serial_compact_without_hidden_copies(
                self,
                *args,
                **kwargs,
            )
        compact_piecewise = getattr(self, "_compact_piecewise_enabled", False)
        forward_context = get_forward_context()
        compact_merged = (
            getattr(self, "_compact_merged_continuations", False)
            and forward_context is not None
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and forward_context.batch_descriptor is not None
            and forward_context.batch_descriptor.num_reqs is not None
        )
        if not compact_piecewise and not compact_merged:
            return original_run_merged_draft(self, *args, **kwargs)

        original_model = self.model
        original_use_cuda_graph = self.use_cuda_graph
        if compact_merged:
            self.model = CompactMergedDraftModel(
                self,
                original_model,
                forward_context.batch_descriptor.num_reqs,
            )
        else:
            self.model = CompactPiecewiseDraftModel(
                self,
                original_model,
                force_piecewise=getattr(
                    self,
                    "_compact_piecewise_force",
                    False,
                ),
            )
            self.use_cuda_graph = False
        try:
            return original_run_merged_draft(self, *args, **kwargs)
        finally:
            self.use_cuda_graph = original_use_cuda_graph
            self.model = original_model

    def sample_aligned_draft(
        self: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        repetition_state = getattr(self, "_vspec_repetition_state", None)
        repetition_prefix = getattr(self, "_vspec_repetition_prefix", None)
        if (
            getattr(self, "_vspec_sparse_proposal_repetition", False)
            and repetition_prefix is not None
        ):
            from .draft_repetition import (
                apply_sparse_draft_repetition_penalty,
            )

            adjusted_logits = apply_sparse_draft_repetition_penalty(
                self,
                logits,
                sampling_metadata,
                repetition_prefix,
            )
            if adjusted_logits is not None:
                token_ids, draft_probs = original_sample_draft(
                    self,
                    adjusted_logits,
                    sampling_metadata,
                )
                repetition_prefix.append(token_ids)
                return token_ids, draft_probs
        if (
            getattr(self, "_vspec_fused_proposal_repetition", False)
            and repetition_prefix is not None
        ):
            from .draft_repetition import fused_draft_repetition_greedy

            token_ids = fused_draft_repetition_greedy(
                self,
                logits,
                sampling_metadata,
                repetition_prefix,
            )
            if token_ids is not None:
                repetition_prefix.append(token_ids)
                return token_ids, None
        if repetition_state is not None and repetition_prefix is not None:
            from .draft_repetition import apply_draft_repetition_penalty

            logits = apply_draft_repetition_penalty(
                logits,
                repetition_state,
                repetition_prefix,
            )
        token_ids, draft_probs = original_sample_draft(
            self,
            logits,
            sampling_metadata,
        )
        if repetition_state is not None and repetition_prefix is not None:
            repetition_prefix.append(token_ids)
        return token_ids, draft_probs

    def compact_attn_update_stack_num_spec_norm(
        self: Any,
        draft_index: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        bound = _bind_host_attn_update(
            original_attn_update,
            self,
            draft_index,
            args,
            kwargs,
        )
        batch_size = bound.arguments["batch_size"]
        aclgraph_runtime_mode = bound.arguments["aclgraph_runtime_mode"]
        if (
            getattr(self, "_vspec_gamma2_fast_metadata", False)
            and draft_index == 1
            and aclgraph_runtime_mode == CUDAGraphMode.FULL
            and self.method == "draft_model"
            and not self.uses_mrope
            and not self.use_compress
            and self.sliding_window is None
            and getattr(self.runner, "dcp_manager", None) is None
            and not getattr(self.runner, "sparse_kv_offload_enabled", False)
        ):
            old_common_metadata = bound.arguments["old_common_metadata"]
            input_batch_size = bound.arguments["input_batch_size"]
            used_update_positions = bound.arguments[
                "used_update_positions"
            ]
            attn_group = bound.arguments["attn_group"]
            common_attn_metadata = self.shallow_copy_metadata(
                old_common_metadata
            )

            common_attn_metadata.num_reqs = input_batch_size
            common_attn_metadata.block_table_tensor = self._adjust_tensor(
                old_common_metadata.block_table_tensor,
                input_batch_size,
            )

            seq_lens = self.seq_lens_group[draft_index][
                :input_batch_size
            ]
            source_seq_lens = old_common_metadata.seq_lens
            copied_reqs = min(input_batch_size, source_seq_lens.shape[0])
            seq_lens[:copied_reqs].copy_(
                source_seq_lens[:copied_reqs]
            )
            if copied_reqs < input_batch_size:
                seq_lens[copied_reqs:].zero_()
            common_attn_metadata.seq_lens = seq_lens

            def copy_cpu_metadata(
                destination: torch.Tensor,
                source: torch.Tensor | None,
            ) -> torch.Tensor | None:
                if source is None:
                    return None
                view = destination[:input_batch_size]
                copied = min(input_batch_size, source.shape[0])
                view[:copied].copy_(source[:copied])
                if copied < input_batch_size:
                    view[copied:].zero_()
                return view

            common_attn_metadata.seq_lens_cpu = copy_cpu_metadata(
                self._vspec_fast_seq_lens_cpu,
                old_common_metadata.seq_lens_cpu,
            )
            common_attn_metadata._seq_lens_cpu = copy_cpu_metadata(
                self._vspec_fast_internal_seq_lens_cpu,
                old_common_metadata._seq_lens_cpu,
            )
            common_attn_metadata.num_computed_tokens_cpu = (
                copy_cpu_metadata(
                    self._vspec_fast_computed_tokens_cpu,
                    old_common_metadata.num_computed_tokens_cpu,
                )
            )

            query_start_loc = self.query_start_loc_group[draft_index][
                : input_batch_size + 1
            ]
            query_start_loc.copy_(self.arange[: input_batch_size + 1])
            common_attn_metadata.query_start_loc = query_start_loc
            common_attn_metadata.query_start_loc_cpu = (
                self._vspec_fast_query_start_loc_cpu[
                    : input_batch_size + 1
                ]
            )
            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.decode_token_per_req = 1
            common_attn_metadata.attn_state = (
                ascend_base_proposer.AscendAttentionState.ChunkedPrefill
            )
            common_attn_metadata.graph_pad_size = -1
            common_attn_metadata.num_input_tokens = input_batch_size

            used_update_positions.add_(1)
            exceeds_max_model_len = (
                used_update_positions >= self.max_model_len
            )
            clamped_positions = torch.where(
                exceeds_max_model_len,
                0,
                used_update_positions,
            )

            seq_lens[:batch_size].add_(1)
            seq_lens[:batch_size].masked_fill_(
                seq_lens[:batch_size] > self.max_model_len,
                1,
            )
            for host_seq_lens in (
                common_attn_metadata.seq_lens_cpu,
                common_attn_metadata._seq_lens_cpu,
            ):
                if host_seq_lens is not None:
                    host_seq_lens[:batch_size].add_(1)
                    host_seq_lens[:batch_size].masked_fill_(
                        host_seq_lens[:batch_size] > self.max_model_len,
                        1,
                    )
            if common_attn_metadata.num_computed_tokens_cpu is not None:
                common_attn_metadata.num_computed_tokens_cpu[
                    :batch_size
                ].add_(1)

            positions = self._vspec_fast_positions
            positions[:input_batch_size].copy_(
                old_common_metadata.positions[:input_batch_size]
            )
            positions[:batch_size].copy_(clamped_positions)
            common_attn_metadata.positions = positions

            block_size = self.block_size
            block_numbers = clamped_positions // block_size
            block_ids = old_common_metadata.block_table_tensor.gather(
                dim=1,
                index=block_numbers.view(-1, 1),
            ).view(-1)
            slot_mapping = (
                block_ids * block_size
                + clamped_positions % block_size
            )
            slot_mapping.masked_fill_(
                exceeds_max_model_len,
                ascend_base_proposer.PADDING_SLOT_ID,
            )
            slot_mapping_group = self.slot_mapping_group[draft_index]
            slot_mapping_group[:batch_size].copy_(
                slot_mapping.to(torch.int32)
            )
            slot_mapping_group[batch_size:].fill_(
                ascend_base_proposer.PADDING_SLOT_ID
            )
            common_attn_metadata.slot_mapping = slot_mapping_group

            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata,
                draft_index,
            )
            return common_attn_metadata, attn_metadata

        precompact_merged = getattr(
            self,
            "_compact_merged_continuations",
            False,
        ) or (
            getattr(self, "_vspec_gamma2_precompact_metadata", False)
            and self.num_speculative_tokens == 2
        )
        if precompact_merged and aclgraph_runtime_mode == CUDAGraphMode.FULL:
            dispatcher = self.runner.cudagraph_dispatcher
            query_len = getattr(
                dispatcher,
                "_vspec_capture_uniform_query_len",
                None,
            ) or getattr(
                dispatcher,
                "_vspec_active_uniform_query_len",
                None,
            )
            input_batch_size = bound.arguments["input_batch_size"]
            if query_len and input_batch_size % query_len == 0:
                bound.arguments["input_batch_size"] = input_batch_size // query_len
        if getattr(self, "_compact_piecewise_enabled", False) and (
            aclgraph_runtime_mode == CUDAGraphMode.PIECEWISE
            or getattr(self, "_compact_piecewise_force", False)
        ):
            if getattr(self, "_compact_piecewise_force", False):
                bound.arguments["aclgraph_runtime_mode"] = CUDAGraphMode.PIECEWISE
                bound.arguments["input_batch_size"] = _compact_piecewise_bucket(self, batch_size)
            else:
                runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                    num_tokens=batch_size,
                    uniform_decode=False,
                    has_lora=False,
                )
                if runtime_mode == CUDAGraphMode.PIECEWISE:
                    bound.arguments["input_batch_size"] = batch_descriptor.num_tokens

        return original_attn_update(*bound.args, **bound.kwargs)

    def draft_graph_update(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if getattr(self, "_compact_piecewise_force", False):
            return None
        return original_draft_graph_update(self, *args, **kwargs)

    def draft_full_graph_update(
        self: Any,
        forward_context: Any,
        num_tokens: int,
        draft_attn_metadatas: list[dict[str, Any]] | None = None,
    ) -> Any:
        unified_compact_num_tokens = getattr(
            self,
            "_vspec_gamma2_unified_compact_num_tokens",
            {},
        ).get(num_tokens)
        if (
            self.num_speculative_tokens == 2
            and getattr(self, "_vspec_gamma2_unified_compact_second", False)
        ):
            descriptor_num_reqs = _uniform_descriptor_request_count(
                getattr(forward_context, "batch_descriptor", None),
                num_tokens,
                self.num_speculative_tokens + 2,
            )
            if descriptor_num_reqs is not None:
                unified_compact_num_tokens = descriptor_num_reqs
        if (
            unified_compact_num_tokens is not None
            and draft_attn_metadatas is not None
            and len(draft_attn_metadatas) > 1
        ):
            continuations_are_compact = all(
                metadata
                and len(
                    next(iter(metadata.values())).actual_seq_lengths_q
                )
                == unified_compact_num_tokens
                for metadata in draft_attn_metadatas[1:]
            )
            if continuations_are_compact:
                compact_metadatas = draft_attn_metadatas
            else:
                compact_metadatas = [draft_attn_metadatas[0]]
                compact_metadatas.extend(
                    _compact_draft_metadata(
                        self,
                        metadata,
                        unified_compact_num_tokens,
                    )
                    for metadata in draft_attn_metadatas[1:]
                )
            return original_draft_full_graph_update(
                self,
                forward_context,
                num_tokens,
                compact_metadatas,
            )

        compact_graph_key = getattr(
            self,
            "_vspec_gamma2_compact_graph_keys",
            {},
        ).get(num_tokens)
        compact_num_tokens = getattr(
            self,
            "_vspec_gamma2_compact_num_tokens",
            {},
        ).get(num_tokens)
        if (
            compact_graph_key is not None
            and compact_num_tokens is not None
            and draft_attn_metadatas is not None
            and len(draft_attn_metadatas) > 1
        ):
            compact_metadata = _compact_draft_metadata(
                self,
                draft_attn_metadatas[1],
                compact_num_tokens,
            )
            if os.environ.get("VSPEC_DRAFT_COMPACT_TRACE") == "1":
                from vllm_ascend.compilation.acl_graph import (
                    get_draft_graph_params,
                )

                graph_params = get_draft_graph_params()
                first_metadata = next(iter(draft_attn_metadatas[0].values()))
                second_metadata = next(iter(compact_metadata.values()))
                _trace_compact_graph(
                    self,
                    "update "
                    f"outer={num_tokens} compact_key={compact_graph_key} "
                    f"outer_handles={len(graph_params.handles[num_tokens])} "
                    f"compact_handles={len(graph_params.handles[compact_graph_key])} "
                    f"q0={first_metadata.actual_seq_lengths_q} "
                    f"q1={second_metadata.actual_seq_lengths_q}",
                )
            original_draft_full_graph_update(
                self,
                forward_context,
                num_tokens,
                draft_attn_metadatas[:1],
            )
            return original_draft_full_graph_update(
                self,
                forward_context,
                compact_graph_key,
                [compact_metadata],
            )
        return original_draft_full_graph_update(
            self,
            forward_context,
            num_tokens,
            draft_attn_metadatas,
        )

    def draft_propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        dispatcher = self.runner.cudagraph_dispatcher
        original_profiled_dispatch = dispatcher.dispatch
        profile_dispatch = bool(os.environ.get("VSPEC_DRAFT_PROFILE_PATH"))
        if profile_dispatch:

            def profiled_dispatch(*dispatch_args: Any, **dispatch_kwargs: Any) -> Any:
                return _profile_dispatch_call(
                    original_profiled_dispatch,
                    *dispatch_args,
                    **dispatch_kwargs,
                )

            dispatcher.dispatch = profiled_dispatch

        bound_propose = inspect.signature(original_draft_propose).bind_partial(
            self,
            *args,
            **kwargs,
        )
        common_attn_metadata = bound_propose.arguments.get("common_attn_metadata")
        batch_size = (
            common_attn_metadata.batch_size()
            if common_attn_metadata is not None
            else settings.max_num_seqs
        )
        repetition_state = getattr(self, "_vspec_repetition_state", None)
        sampling_metadata = bound_propose.arguments.get("sampling_metadata")
        if repetition_state is not None and sampling_metadata is not None:
            input_batch = self.runner.input_batch
            repetition_state.update(
                prompt_token_ids=input_batch.token_ids_cpu,
                prompt_lengths=input_batch.num_prompt_tokens,
                output_token_ids=sampling_metadata.output_token_ids,
                repetition_penalties=input_batch.repetition_penalties_cpu,
                num_rows=batch_size,
                vocab_size=self.draft_model_config.get_vocab_size(),
            )
            self._vspec_repetition_prefix.clear()
        hybrid_state = getattr(self, "_vspec_hybrid_second_token", None)
        if hybrid_state is not None:
            hybrid_state.prepare(self.runner.input_batch, batch_size)
        use_merged_full = settings.use_merged_full or (
            settings.merged_full_max_batch > 0 and batch_size <= settings.merged_full_max_batch
        )
        had_active_query_len = hasattr(
            dispatcher,
            "_vspec_active_uniform_query_len",
        )
        previous_active_query_len = getattr(
            dispatcher,
            "_vspec_active_uniform_query_len",
            None,
        )
        had_active_num_reqs = hasattr(
            dispatcher,
            "_vspec_active_num_reqs",
        )
        previous_active_num_reqs = getattr(
            dispatcher,
            "_vspec_active_num_reqs",
            None,
        )
        had_compact_first_pass = hasattr(
            dispatcher,
            "_vspec_compact_first_pass",
        )
        previous_compact_first_pass = getattr(
            dispatcher,
            "_vspec_compact_first_pass",
            None,
        )
        if use_merged_full:
            dispatcher._vspec_active_num_reqs = batch_size
            runtime_query_len = getattr(
                dispatcher,
                "_vspec_runtime_draft_query_len",
                None,
            )
            if runtime_query_len is not None:
                dispatcher._vspec_active_uniform_query_len = runtime_query_len
            elif hasattr(dispatcher, "_vspec_active_uniform_query_len"):
                del dispatcher._vspec_active_uniform_query_len

        def restore_dispatcher() -> None:
            dispatcher.dispatch = original_profiled_dispatch
            if had_active_query_len:
                dispatcher._vspec_active_uniform_query_len = previous_active_query_len
            elif hasattr(dispatcher, "_vspec_active_uniform_query_len"):
                del dispatcher._vspec_active_uniform_query_len
            if had_active_num_reqs:
                dispatcher._vspec_active_num_reqs = previous_active_num_reqs
            elif hasattr(dispatcher, "_vspec_active_num_reqs"):
                del dispatcher._vspec_active_num_reqs
            if had_compact_first_pass:
                dispatcher._vspec_compact_first_pass = previous_compact_first_pass
            elif hasattr(dispatcher, "_vspec_compact_first_pass"):
                del dispatcher._vspec_compact_first_pass

        force_compact = (
            use_merged_full
            and not kwargs.get("num_prefill_reqs", 0)
            and os.environ.get("VSPEC_DRAFT_COMPACT_CONTINUATIONS") == "1"
        )
        if force_compact:
            previous_padding = getattr(
                self.runner,
                "_draft_merged_full_padding",
                False,
            )
            previous_runnable = self._runnable
            self.runner._draft_merged_full_padding = True
            self._compact_piecewise_enabled = True
            self._compact_piecewise_force = True
            self._runnable = self._run_merged_draft
            try:
                result = original_draft_propose(self, *args, **kwargs)
            finally:
                self._runnable = previous_runnable
                self._compact_piecewise_force = False
                self._compact_piecewise_enabled = False
                self.runner._draft_merged_full_padding = previous_padding
                restore_dispatcher()
            _record_draft_profile(
                "draft_propose",
                (time.perf_counter() - started_at) * 1000.0,
            )
            return result

        if use_merged_full:
            previous_padding = getattr(self.runner, "_draft_merged_full_padding", False)
            self.runner._draft_merged_full_padding = True
            try:
                result = original_draft_propose(self, *args, **kwargs)
            finally:
                self.runner._draft_merged_full_padding = previous_padding
                restore_dispatcher()
            _record_draft_profile(
                "draft_propose",
                (time.perf_counter() - started_at) * 1000.0,
            )
            return result

        original_dispatch = dispatcher.dispatch

        def dispatch_nonuniform(*dispatch_args: Any, **dispatch_kwargs: Any) -> Any:
            dispatch_kwargs["uniform_decode"] = False
            return original_dispatch(*dispatch_args, **dispatch_kwargs)

        dispatcher.dispatch = dispatch_nonuniform
        self._compact_piecewise_enabled = True
        try:
            result = original_draft_propose(self, *args, **kwargs)
        finally:
            self._compact_piecewise_enabled = False
            restore_dispatcher()
        _record_draft_profile(
            "draft_propose",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    AscendDraftModelProposer.__init__ = init_draft_proposer
    ascend_base_proposer.AscendSpecDecodeBaseProposer.dummy_run = draft_dummy_run
    NPUModelRunner._pad_query_start_loc_for_fia = pad_query_start_loc_for_fia
    ascend_base_proposer.compute_new_slot_mapping = compute_draft_slot_mapping
    ascend_base_proposer.AscendSpecDecodeBaseProposer.set_inputs_first_pass = (
        compact_first_pass_inputs
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer._run_merged_draft = (
        run_compact_piecewise_draft
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm = (
        compact_attn_update_stack_num_spec_norm
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer._update_full_graph_params_if_needed = (
        draft_graph_update
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer._update_full_graph_params = (
        draft_full_graph_update
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer._sample_draft_from_logits = (
        sample_aligned_draft
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer._propose = draft_propose
    setattr(AscendDraftModelProposer, PATCH_MARKER, True)
    return True
