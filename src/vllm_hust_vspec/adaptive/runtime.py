"""Runtime hooks for vSpec Adaptive scheduling and dynamic Draft lengths."""

from __future__ import annotations

import logging
import os
import time
from bisect import bisect_left
from collections import deque
from contextlib import contextmanager
from dataclasses import fields, replace
from typing import Any

from ..config import PluginSettings
from ..host_compat import call_with_supported_kwargs
from .controller import GoodputController
from .entropy import EntropyDraftStopper, topk_entropy
from .online import OnlineGammaController
from .profile import AdaptiveProfile

logger = logging.getLogger(__name__)

# Gamma 1 has no suffix to trim. At gamma 2, the first-position confidence
# already determines whether verifying the second proposal is worthwhile.
_MIN_CONFIDENCE_STOP_GAMMA = 2

SCHEDULER_PATCH_MARKER = "_vllm_hust_vspec_adaptive_scheduler_patched"
PROPOSER_PATCH_MARKER = "_vllm_hust_vspec_adaptive_proposer_patched"
RUNNER_PATCH_MARKER = "_vllm_hust_vspec_adaptive_runner_patched"
CAPTURE_SIZE_PATCH_MARKER = "_vllm_hust_vspec_adaptive_capture_sizes_patched"
DISPATCHER_PATCH_MARKER = "_vllm_hust_vspec_adaptive_dispatcher_patched"
ASYNC_OUTPUT_PATCH_MARKER = "_vllm_hust_vspec_adaptive_output_patched"
_EAGLE_UNIFORM_STATE_ENV = "VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL"
_EAGLE_DYNAMIC_UNIFORM_STATE_ENV = "HUST_VSPEC_EAGLE_DYNAMIC_UNIFORM_STATE_KERNEL"
_EAGLE_CONFIDENCE_ACCEPT_ATTR = "_vspec_eagle_confidence_accept_enabled"


class _TargetOnlyLatch:
    """Keep Draft-disabled requests sticky until their cohort drains."""

    def __init__(self) -> None:
        self.request_ids: set[str] = set()

    @property
    def active(self) -> bool:
        return bool(self.request_ids)

    def refresh(self, active_request_ids: set[str]) -> bool:
        if not self.request_ids:
            return False
        self.request_ids.intersection_update(active_request_ids)
        if not self.request_ids:
            return False
        self.request_ids.update(active_request_ids)
        return True

    def enter(self, active_request_ids: set[str]) -> None:
        self.request_ids.update(active_request_ids)


def _disable_fixed_width_eagle_state_kernel(
    method: str,
    min_gamma: int,
    max_gamma: int,
    environ: Any,
) -> bool:
    """Disable an upstream fixed-width state shortcut for dynamic EAGLE."""
    if method == "eagle" and min_gamma < max_gamma and environ.get(_EAGLE_UNIFORM_STATE_ENV) == "1":
        environ[_EAGLE_DYNAMIC_UNIFORM_STATE_ENV] = "1"
        environ[_EAGLE_UNIFORM_STATE_ENV] = "0"
        return True
    return False


def _context_tokens_after_schedule(scheduler: Any, scheduler_output: Any) -> int:
    total = 0
    for request_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests.get(request_id)
        computed_tokens = int(getattr(request, "num_computed_tokens", 0))
        total += max(1, computed_tokens + int(scheduled_tokens))
    return total


def _adaptive_scheduled_batch_size(
    scheduler: Any,
    num_scheduled_requests: int,
) -> int:
    """Resolve the controller batch size across scheduler API versions."""
    get_batch_size = getattr(scheduler, "_get_dynamic_sd_batch_size", None)
    if get_batch_size is not None:
        return int(get_batch_size(num_scheduled_requests))
    return int(num_scheduled_requests)


def _adaptive_target_query_lens(existing: Any, max_gamma: int) -> tuple[int, ...]:
    widths = {int(width) for width in existing}
    widths.update(range(1, max_gamma + 2))
    return tuple(sorted(widths))


def _uses_width_isolated_target_graph_params(cudagraph_mode: Any) -> bool:
    """Return whether Target graph state is independent for each query width."""
    return getattr(cudagraph_mode, "name", "") == "FULL_DECODE_ONLY"


def _runtime_target_query_width(scheduler_output: Any) -> int | None:
    scheduled = scheduler_output.num_scheduled_tokens
    draft_tokens = scheduler_output.scheduled_spec_decode_tokens
    widths = {len(draft_tokens.get(request_id, ())) + 1 for request_id in scheduled}
    return widths.pop() if len(widths) == 1 else None


def _current_target_is_target_only(
    current_query_width: int | None,
    proposal_gamma: int,
) -> bool:
    """Identify the Target frame independently of the next proposal width."""
    if current_query_width is not None:
        return current_query_width == 1
    return proposal_gamma == 0


def _current_eagle_confidence_accept_enabled(
    current_query_width: int | None,
    proposal_gamma: int,
) -> bool:
    """Use relaxed EAGLE verification only for the validated gamma range."""
    current_gamma = current_query_width - 1 if current_query_width is not None else proposal_gamma
    return current_gamma <= 2


@contextmanager
def _runtime_dynamic_eagle_state_kernel(
    runner: Any,
    *,
    stable_decode: bool,
    gamma: int,
    environ: Any = os.environ,
) -> Any:
    """Enable the native state kernel only at its stable anchor width."""
    if environ.get(_EAGLE_DYNAMIC_UNIFORM_STATE_ENV) != "1":
        yield
        return

    previous_enabled = environ.get(_EAGLE_UNIFORM_STATE_ENV)
    enabled = stable_decode and runner.num_spec_tokens == gamma
    environ[_EAGLE_UNIFORM_STATE_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous_enabled is None:
            environ.pop(_EAGLE_UNIFORM_STATE_ENV, None)
        else:
            environ[_EAGLE_UNIFORM_STATE_ENV] = previous_enabled


def _batch_descriptor_query_width(batch_descriptor: Any) -> int | None:
    """Return the uniform query width represented by a graph descriptor."""
    if batch_descriptor is None or not getattr(
        batch_descriptor,
        "uniform",
        False,
    ):
        return None
    num_tokens = int(getattr(batch_descriptor, "num_tokens", 0))
    num_reqs = int(getattr(batch_descriptor, "num_reqs", 0) or 0)
    if num_tokens <= 0 or num_reqs <= 0 or num_tokens % num_reqs:
        return None
    return num_tokens // num_reqs


def _proposal_input_query_width(
    common_attn_metadata: Any,
    batch_descriptor: Any,
) -> int | None:
    """Resolve the Target output width consumed by the Draft first pass."""
    # Uniform descriptors already encode the exact logical width. Reading them
    # first avoids materializing a per-request tensor as a Python list on every
    # stable decode step.
    descriptor_width = _batch_descriptor_query_width(batch_descriptor)
    if descriptor_width is not None:
        return descriptor_width
    if common_attn_metadata is not None:
        query_start_loc = getattr(
            common_attn_metadata,
            "query_start_loc_cpu",
            None,
        )
        if query_start_loc is not None:
            starts = [int(value) for value in query_start_loc.tolist()]
            adjacent_starts = zip(starts, starts[1:], strict=False)
            widths = {end - start for start, end in adjacent_starts}
            if len(widths) == 1:
                width = widths.pop()
                if width > 0:
                    return width
    return None


def _nonuniform_batch_descriptor(batch_descriptor: Any) -> Any:
    if batch_descriptor is None or not getattr(
        batch_descriptor,
        "uniform",
        False,
    ):
        return batch_descriptor
    return replace(batch_descriptor, uniform=False)


def _uniform_decode_fits_capture_bucket(
    dispatcher: Any,
    num_tokens: int,
    query_width: int,
) -> bool:
    padded_sizes = getattr(dispatcher, "_bs_to_padded_graph_size", ())
    if query_width <= 0 or num_tokens < 0 or num_tokens >= len(padded_sizes):
        return True
    return int(padded_sizes[num_tokens]) % query_width == 0


def _add_adaptive_decode_graph_keys(
    dispatcher: Any,
    query_widths: tuple[int, ...],
) -> None:
    """Add FULL_DECODE_ONLY descriptors for every reachable runtime width."""
    from vllm.config import CUDAGraphMode

    if not dispatcher.cudagraph_mode.separate_routine():
        return
    capture_sizes = dispatcher.compilation_config.cudagraph_capture_sizes or ()
    max_num_seqs = dispatcher.vllm_config.scheduler_config.max_num_seqs
    lora_cases = dispatcher._get_lora_cases()
    previous_width = dispatcher.uniform_decode_query_len
    try:
        for query_width in query_widths:
            dispatcher.uniform_decode_query_len = query_width
            for size in capture_sizes:
                if size < query_width or size > query_width * max_num_seqs or size % query_width:
                    continue
                for num_active_loras in lora_cases:
                    descriptor = dispatcher._create_padded_batch_descriptor(
                        size,
                        True,
                        num_active_loras > 0,
                        num_active_loras,
                    )
                    dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, descriptor)
    finally:
        dispatcher.uniform_decode_query_len = previous_width


def _initialize_adaptive_decode_only_graph_keys(
    dispatcher: Any,
    cudagraph_mode: Any,
    query_widths: tuple[int, ...],
) -> None:
    """Initialize FULL_DECODE_ONLY keys without mixing verification widths."""
    dispatcher.cudagraph_mode = cudagraph_mode
    dispatcher._compute_bs_to_padded_graph_size()
    lora_cases = dispatcher._get_lora_cases()
    dispatcher.captured_lora_counts = [count for count in lora_cases if count]
    _add_adaptive_decode_graph_keys(dispatcher, query_widths)
    dispatcher.keys_initialized = True


@contextmanager
def _runner_query_width(
    runner: Any,
    query_width: int | None,
) -> Any:
    """Temporarily align runner and dispatcher with one Target frame width."""
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    if runner is None or dispatcher is None or query_width is None:
        yield
        return

    previous_runner_width = runner.uniform_decode_query_len
    previous_dispatcher_width = dispatcher.uniform_decode_query_len
    if previous_runner_width == query_width and previous_dispatcher_width == query_width:
        yield
        return
    runner.uniform_decode_query_len = query_width
    dispatcher.uniform_decode_query_len = query_width
    try:
        yield
    finally:
        runner.uniform_decode_query_len = previous_runner_width
        dispatcher.uniform_decode_query_len = previous_dispatcher_width


@contextmanager
def _runtime_runner_query_width(
    proposer: Any,
    query_width: int | None,
) -> Any:
    """Temporarily align shared runner state with the current Target frame."""
    with _runner_query_width(getattr(proposer, "runner", None), query_width):
        yield


@contextmanager
def _runtime_eager_dispatch(proposer: Any, enabled: bool) -> Any:
    """Prevent graph dispatch when the selected Draft body is eager."""
    runner = getattr(proposer, "runner", None)
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    if dispatcher is None or not enabled:
        yield
        return

    previous_dispatch = dispatcher.dispatch

    def dispatch_nonuniform(*args: Any, **kwargs: Any) -> Any:
        kwargs["uniform_decode"] = False
        return previous_dispatch(*args, **kwargs)

    dispatcher.dispatch = dispatch_nonuniform
    try:
        yield
    finally:
        dispatcher.dispatch = previous_dispatch


@contextmanager
def _runtime_eager_proposer(proposer: Any, enabled: bool) -> Any:
    """Bypass proposer graph setup for a shape-incompatible transition."""
    if not enabled:
        yield
        return
    previous_use_cuda_graph = proposer.use_cuda_graph
    proposer.use_cuda_graph = False
    try:
        yield
    finally:
        proposer.use_cuda_graph = previous_use_cuda_graph


def _update_async_next_frame_gamma(
    scheduler: Any,
    scheduler_output: Any,
    gamma: int,
) -> None:
    """Repair AsyncScheduler state after a post-schedule gamma decision."""
    scheduler_config = getattr(scheduler, "scheduler_config", None)
    if not getattr(scheduler_config, "async_scheduling", False):
        return

    placeholders = [-1] * gamma
    scheduler._spec_token_placeholders = placeholders
    for request_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(request_id)
        if request is None or getattr(request, "is_prefill_chunk", False):
            continue
        # AsyncScheduler already ran _update_after_schedule with the old gamma.
        # Replace only the next-frame Draft placeholders; current-frame
        # accounting remains tied to scheduled_spec_decode_tokens.
        request.spec_token_ids = placeholders


def _run_with_runtime_gamma(
    proposer: Any,
    requested_gamma: int,
    callback: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    previous_gamma = proposer.num_speculative_tokens
    previous_steps = getattr(proposer, "num_draft_steps", previous_gamma)
    previous_threshold = getattr(proposer, "decode_threshold", previous_gamma + 1)
    previous_extra_slots = getattr(proposer, "extra_slots_per_request", None)
    previous_net_slots = getattr(
        proposer,
        "net_num_new_slots_per_request",
        None,
    )
    previous_needs_extra_slots = getattr(
        proposer,
        "needs_extra_input_slots",
        None,
    )
    proposer.num_speculative_tokens = requested_gamma
    proposer.num_draft_steps = requested_gamma
    proposer.decode_threshold = requested_gamma + 1
    if getattr(proposer, "parallel_drafting", False):
        proposer.extra_slots_per_request = requested_gamma
        proposer.net_num_new_slots_per_request = requested_gamma - (
            1
            if (
                getattr(proposer, "pass_hidden_states_to_model", False)
                and getattr(proposer, "method", None) != "dflash"
            )
            else 0
        )
        proposer.needs_extra_input_slots = proposer.net_num_new_slots_per_request > 0
    try:
        return callback(proposer, *args, **kwargs)
    finally:
        proposer.num_speculative_tokens = previous_gamma
        proposer.num_draft_steps = previous_steps
        proposer.decode_threshold = previous_threshold
        if previous_extra_slots is not None:
            proposer.extra_slots_per_request = previous_extra_slots
        if previous_net_slots is not None:
            proposer.net_num_new_slots_per_request = previous_net_slots
        if previous_needs_extra_slots is not None:
            proposer.needs_extra_input_slots = previous_needs_extra_slots


def _proposal_execution_gamma(
    method: str | None,
    requested_gamma: int,
    previous_gamma: int,
) -> int:
    """Keep serial EAGLE aligned while reducing the proposal width."""
    if method == "eagle" and 0 < requested_gamma < previous_gamma:
        return previous_gamma
    return requested_gamma


def _copy_dynamic_draft_tokens(
    destination: Any,
    source: Any,
    *,
    zeros_only: bool = False,
) -> None:
    num_reqs, num_spec_tokens = source.shape[:2]
    if num_spec_tokens > destination.shape[1]:
        raise RuntimeError(
            "dynamic Draft output exceeds the configured D2H buffer width: "
            f"runtime={num_spec_tokens}, configured={destination.shape[1]}"
        )
    active = destination[:num_reqs, :num_spec_tokens]
    if zeros_only:
        active.zero_()
    else:
        active.copy_(source, non_blocking=True)
    if num_spec_tokens < destination.shape[1]:
        destination[:num_reqs, num_spec_tokens:].fill_(-1)


def _trim_placeholder_suffixes(draft_token_ids: Any) -> Any:
    """Convert fixed-width device output rows into scheduler-visible lengths."""
    for row in draft_token_ids.draft_token_ids:
        try:
            first_placeholder = row.index(-1)
        except ValueError:
            continue
        del row[first_placeholder:]
    return draft_token_ids


def _unwrap_model_runner_output(result: Any) -> Any:
    output = getattr(result, "model_runner_output", None)
    if output is None:
        output = getattr(result, "_model_runner_output", result)
    return output


def _initialize_full_graph_runnables(
    proposer: Any,
    configured_gamma: int,
    min_gamma: int,
) -> dict[int, Any] | None:
    from vllm.config import CUDAGraphMode
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    existing = getattr(
        proposer,
        "_vspec_adaptive_full_graph_runnables",
        None,
    )
    if existing is not None:
        return existing

    runnable = proposer._runnable
    if (
        not proposer.use_cuda_graph
        or not isinstance(runnable, ACLGraphWrapper)
        or runnable.runtime_mode != CUDAGraphMode.FULL
    ):
        return None

    graph_min_gamma = max(1, min_gamma)
    body = runnable.unwrap()
    runnables = {configured_gamma: runnable}
    for gamma in range(configured_gamma - 1, graph_min_gamma - 1, -1):
        runnables[gamma] = call_with_supported_kwargs(
            ACLGraphWrapper,
            body,
            proposer.vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=runnable.aclgraph_options,
            use_eagle=runnable.use_eagle,
            enable_enpu=runnable.enable_enpu,
            is_draft_model=True,
        )
    proposer._vspec_adaptive_full_graph_runnables = runnables
    proposer._vspec_adaptive_full_graph_body = body
    return runnables


def _empty_graph_params_like(graph_params: Any) -> Any:
    """Create an isolated graph-parameter table with identical shape keys."""
    graph_params_type = type(graph_params)
    sizes = tuple(graph_params.events)
    values = {}
    for field in fields(graph_params):
        source = getattr(graph_params, field.name)
        if not isinstance(source, dict):
            raise TypeError(f"unsupported graph parameter field: {field.name}")
        if field.name == "workspaces":
            values[field.name] = {size: None for size in sizes}
        else:
            values[field.name] = {size: [] for size in sizes}
    return graph_params_type(**values)


def _install_draft_entropy_probe(proposer: Any) -> None:
    """Capture per-request entropy inside each Draft logits operation."""
    if getattr(proposer, "_vspec_entropy_probe_installed", False):
        return
    model = proposer.model
    original_compute_logits = model.compute_logits

    def compute_logits(*args: Any, **kwargs: Any) -> Any:
        result = original_compute_logits(*args, **kwargs)
        if not getattr(proposer, "_vspec_entropy_measure_enabled", True):
            return result
        raw_logits = getattr(result, "logits", result)
        try:
            import torch

            if torch.is_tensor(raw_logits):
                entropy = topk_entropy(
                    raw_logits,
                    int(proposer._vspec_entropy_topk),
                )
                proposer._vspec_entropy_recent_values.append(entropy)
        except AttributeError:
            pass
        return result

    proposer._vspec_entropy_recent_values = []
    proposer._vspec_entropy_graph_values = {}
    proposer._vspec_entropy_original_compute_logits = original_compute_logits
    model.compute_logits = compute_logits
    proposer._vspec_entropy_probe_installed = True


def _draft_entropy_matrix(
    proposer: Any,
    gamma: int,
    batch_size: int,
    topk: int,
) -> Any | None:
    """Read graph-stable current-round entropy as [position, request]."""
    proposer._vspec_entropy_topk = topk
    recent = list(getattr(proposer, "_vspec_entropy_recent_values", ()))
    if len(recent) >= gamma:
        entropy_by_position = recent[-gamma:]
    else:
        captured_by_batch = getattr(
            proposer,
            "_vspec_entropy_graph_values",
            {},
        ).get(gamma, {})
        captured_batch = next(
            (size for size in sorted(captured_by_batch) if size >= batch_size),
            None,
        )
        entropy_by_position = (
            list(captured_by_batch[captured_batch]) if captured_batch is not None else []
        )
    if len(entropy_by_position) < gamma:
        return None
    import torch

    return torch.stack(
        [value[:batch_size] for value in entropy_by_position[:gamma]],
        dim=0,
    )


def _prepare_gamma_graph_params(
    proposer: Any,
    configured_gamma: int,
    min_gamma: int,
    base_graph_params: Any,
) -> dict[int, Any] | None:
    if base_graph_params is None:
        return None
    source_id = id(base_graph_params)
    if getattr(proposer, "_vspec_adaptive_graph_params_source_id", None) != source_id:
        graph_min_gamma = max(1, min_gamma)
        proposer._vspec_adaptive_graph_params_by_gamma = {
            gamma: (
                base_graph_params
                if gamma == configured_gamma
                else _empty_graph_params_like(base_graph_params)
            )
            for gamma in range(graph_min_gamma, configured_gamma + 1)
        }
        proposer._vspec_adaptive_graph_params_source_id = source_id
    return proposer._vspec_adaptive_graph_params_by_gamma


def _prepare_target_graph_params(
    runner: Any,
    base_graph_params: Any,
    query_widths: tuple[int, ...],
) -> dict[int, Any] | None:
    if base_graph_params is None:
        return None
    source_id = id(base_graph_params)
    if getattr(runner, "_vspec_adaptive_target_graph_source_id", None) != source_id:
        base_width = max(query_widths)
        runner._vspec_adaptive_target_graph_params_by_width = {
            width: (
                base_graph_params
                if width == base_width
                else _empty_graph_params_like(base_graph_params)
            )
            for width in query_widths
        }
        runner._vspec_adaptive_target_graph_source_id = source_id
    return runner._vspec_adaptive_target_graph_params_by_width


def _get_full_graph_runnable_for_gamma(
    proposer: Any,
    gamma: int,
    configured_gamma: int,
    min_gamma: int,
) -> Any:
    runnables = _initialize_full_graph_runnables(
        proposer,
        configured_gamma,
        min_gamma,
    )
    if runnables is None:
        return proposer._runnable
    return runnables[gamma]


def _capture_all_adaptive_draft_graphs(
    proposer: Any,
    callback: Any,
    configured_gamma: int,
    min_gamma: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Warm and capture one FULL Draft graph per stable runtime gamma."""
    runnables = _initialize_full_graph_runnables(
        proposer,
        configured_gamma,
        min_gamma,
    )
    if runnables is None:
        return callback(proposer, *args, **kwargs)

    from vllm_ascend.compilation import acl_graph as acl_graph_module

    base_graph_params = acl_graph_module._draft_graph_params
    graph_params_by_gamma = _prepare_gamma_graph_params(
        proposer,
        configured_gamma,
        min_gamma,
        base_graph_params,
    )

    previous_runnable = proposer._runnable
    previous_entropy_enabled = getattr(
        proposer,
        "_vspec_entropy_measure_enabled",
        True,
    )
    configured_result = None
    try:
        # Capture the largest graph first so smaller graphs reuse its pool.
        graph_min_gamma = max(1, min_gamma)
        for gamma in range(configured_gamma, graph_min_gamma - 1, -1):
            proposer._vspec_entropy_measure_enabled = gamma >= _MIN_CONFIDENCE_STOP_GAMMA
            if getattr(proposer, "_vspec_entropy_probe_installed", False):
                proposer._vspec_entropy_recent_values = []
            proposer._runnable = runnables[gamma]
            if graph_params_by_gamma is not None:
                acl_graph_module._draft_graph_params = graph_params_by_gamma[gamma]
            result = _run_with_runtime_gamma(
                proposer,
                gamma,
                callback,
                *args,
                **kwargs,
            )
            if getattr(proposer, "_vspec_entropy_probe_installed", False):
                recent = proposer._vspec_entropy_recent_values
                if len(recent) >= gamma:
                    captured = tuple(recent[-gamma:])
                    captured_batch = int(captured[0].shape[0])
                    proposer._vspec_entropy_graph_values.setdefault(
                        gamma,
                        {},
                    )[captured_batch] = captured
            if gamma == configured_gamma:
                configured_result = result
    finally:
        proposer._vspec_entropy_measure_enabled = previous_entropy_enabled
        proposer._runnable = previous_runnable
        acl_graph_module._draft_graph_params = base_graph_params
    return configured_result


def _full_graph_batch_matches_capture(
    common_attn_metadata: Any,
    target_model_batch_desc: Any,
    gamma: int,
    capture_sizes: list[int] | None = None,
) -> bool:
    if common_attn_metadata is None or target_model_batch_desc is None:
        return False
    num_tokens = int(getattr(target_model_batch_desc, "num_tokens", 0))
    if num_tokens <= 0:
        return False
    actual_batch = int(common_attn_metadata.batch_size())
    actual_tokens = int(
        getattr(
            common_attn_metadata,
            "num_actual_tokens",
            actual_batch * (gamma + 1),
        )
    )
    if actual_tokens < actual_batch * (gamma + 1):
        return False
    if capture_sizes:
        index = bisect_left(capture_sizes, actual_tokens)
        return index < len(capture_sizes) and num_tokens == capture_sizes[index]
    return num_tokens == actual_tokens


def apply_adaptive_patches(settings: PluginSettings) -> bool:
    """Install scheduler feedback and Ascend dynamic-gamma runtime hooks."""
    if not settings.adaptive_speculation:
        return False

    if _disable_fixed_width_eagle_state_kernel(
        settings.method,
        settings.adaptive_min_gamma,
        settings.adaptive_max_gamma,
        os.environ,
    ):
        logger.warning(
            "%s was isolated from dynamic EAGLE transitions; it is only "
            "enabled for stable frames at the runtime state width",
            _EAGLE_UNIFORM_STATE_ENV,
        )

    from vllm.config import CUDAGraphMode
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )
    from vllm_ascend.worker import model_runner_v1 as model_runner_module
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    AscendDflashProposer = None
    if settings.method == "dflash":
        from vllm_ascend.spec_decode.dflash_proposer import (
            AscendDflashProposer,
        )

    profile = None
    if settings.adaptive_policy == "profile":
        profile = AdaptiveProfile.load(settings.adaptive_profile_path)
        if settings.adaptive_max_gamma > profile.max_speculative_tokens:
            raise RuntimeError(
                "vSpec Adaptive max gamma exceeds profile capacity: "
                f"configured={settings.adaptive_max_gamma}, "
                f"profile={profile.max_speculative_tokens}"
            )
        expected_parallel_profile = settings.method == "dflash"
        if profile.draft_parallel != expected_parallel_profile:
            raise RuntimeError(
                "vSpec Adaptive profile draft_parallel does not match method: "
                f"method={settings.method}, profile={profile.draft_parallel}"
            )

    # Online control measures realized output cadence at the scheduler. Model
    # execution intervals overlap under async scheduling and cannot be summed
    # into serving goodput.
    measure_step_latency = (
        settings.adaptive_latency_calibration and settings.adaptive_policy != "online"
    )
    draft_stopper = (
        EntropyDraftStopper(
            max_gamma=settings.adaptive_max_gamma,
            min_gamma=settings.adaptive_min_gamma,
            threshold=settings.adaptive_entropy_threshold,
            scale=settings.adaptive_entropy_scale,
        )
        if settings.adaptive_entropy_stop
        else None
    )

    applied = False
    if measure_step_latency and not getattr(
        AsyncGPUModelRunnerOutput,
        ASYNC_OUTPUT_PATCH_MARKER,
        False,
    ):
        original_async_get_output = AsyncGPUModelRunnerOutput.get_output

        def adaptive_async_get_output(self: Any) -> Any:
            output = original_async_get_output(self)
            measurement = getattr(
                output,
                "vspec_adaptive_step_measurement",
                None,
            )
            if measurement and "started_at" in measurement:
                started_at = float(measurement.pop("started_at"))
                measurement["latency_ms"] = (time.perf_counter() - started_at) * 1000.0
                if settings.adaptive_trace:
                    logger.warning(
                        "vSpec Adaptive async latency completed: gamma=%d batch=%d latency_ms=%.4f",
                        measurement["gamma"],
                        measurement["batch_size"],
                        measurement["latency_ms"],
                    )
            return output

        AsyncGPUModelRunnerOutput.get_output = adaptive_async_get_output
        setattr(
            AsyncGPUModelRunnerOutput,
            ASYNC_OUTPUT_PATCH_MARKER,
            True,
        )
        applied = True
    if not getattr(CudagraphDispatcher, DISPATCHER_PATCH_MARKER, False):
        original_dispatch = CudagraphDispatcher.dispatch

        def adaptive_dispatch(
            self: Any,
            num_tokens: int,
            uniform_decode: bool = False,
            has_lora: bool = False,
            num_active_loras: int = 0,
            valid_modes: Any = None,
            invalid_modes: Any = None,
            uniform_decode_query_len: int | None = None,
        ) -> Any:
            query_width = int(
                uniform_decode_query_len
                if uniform_decode_query_len is not None
                else self.uniform_decode_query_len
            )
            if uniform_decode and not _uniform_decode_fits_capture_bucket(
                self,
                num_tokens,
                query_width,
            ):
                uniform_decode = False
            return call_with_supported_kwargs(
                original_dispatch,
                self,
                num_tokens=num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
                valid_modes=valid_modes,
                invalid_modes=invalid_modes,
                uniform_decode_query_len=uniform_decode_query_len,
            )

        CudagraphDispatcher.dispatch = adaptive_dispatch
        setattr(CudagraphDispatcher, DISPATCHER_PATCH_MARKER, True)
        applied = True

    original_restore_capture_sizes = getattr(
        model_runner_module,
        "_restore_nanoparl_dynamic_graph_capture_sizes",
        None,
    )
    if not getattr(
        model_runner_module,
        CAPTURE_SIZE_PATCH_MARKER,
        False,
    ):

        def restore_adaptive_capture_sizes(
            compilation_config: Any,
            *,
            method: str | None,
            requested_capture_sizes: list[int],
        ) -> bool:
            if settings.adaptive_full_graph and requested_capture_sizes:
                exact_sizes = sorted(set(int(size) for size in requested_capture_sizes))
                if any(size <= 0 for size in exact_sizes):
                    raise ValueError("vSpec Adaptive graph capture sizes must be positive")
                compilation_config.cudagraph_capture_sizes = exact_sizes
                compilation_config.max_cudagraph_capture_size = exact_sizes[-1]
                return True
            if original_restore_capture_sizes is None:
                return False
            return original_restore_capture_sizes(
                compilation_config,
                method=method,
                requested_capture_sizes=requested_capture_sizes,
            )

        model_runner_module._restore_nanoparl_dynamic_graph_capture_sizes = (
            restore_adaptive_capture_sizes
        )
        setattr(
            model_runner_module,
            CAPTURE_SIZE_PATCH_MARKER,
            True,
        )
        applied = True

    if not getattr(Scheduler, SCHEDULER_PATCH_MARKER, False):
        original_scheduler_init = Scheduler.__init__
        original_schedule = Scheduler.schedule
        original_make_stats = Scheduler.make_spec_decoding_stats
        original_update_from_output = Scheduler.update_from_output
        original_update_draft_token_ids = Scheduler.update_draft_token_ids

        def scheduler_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_scheduler_init(self, *args, **kwargs)
            if self.scheduler_config.async_scheduling and not settings.adaptive_async:
                raise RuntimeError("vSpec Adaptive does not yet support async scheduling")
            graph_mode = self.vllm_config.compilation_config.cudagraph_mode
            allowed_graph_modes = {CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE}
            if settings.adaptive_full_graph:
                allowed_graph_modes.update(
                    {
                        CUDAGraphMode.FULL,
                        CUDAGraphMode.FULL_DECODE_ONLY,
                        CUDAGraphMode.FULL_AND_PIECEWISE,
                    }
                )
            if graph_mode not in allowed_graph_modes:
                raise RuntimeError(
                    f"vSpec Adaptive requires Eager or PIECEWISE graph mode, got {graph_mode}"
                )
            common_options = {
                "max_gamma": settings.adaptive_max_gamma,
                "min_gamma": settings.adaptive_min_gamma,
                "ewma_weight": settings.adaptive_ewma_weight,
                "control_interval": settings.adaptive_control_interval,
                "hysteresis": settings.adaptive_hysteresis,
                "max_gamma_step": settings.adaptive_max_gamma_step,
                "max_batched_tokens": int(self.scheduler_config.max_num_batched_tokens),
            }
            if settings.adaptive_policy == "online":
                self._vspec_adaptive_controller = OnlineGammaController(
                    **common_options,
                    window_size=settings.adaptive_online_window,
                    exploration=settings.adaptive_online_exploration,
                    warmup_samples=(settings.adaptive_online_warmup_samples),
                    warmup_return=(settings.adaptive_online_warmup_return),
                    inflight_warmup_padding=(1 if settings.adaptive_async else 0),
                )
            else:
                assert profile is not None
                self._vspec_adaptive_controller = GoodputController(
                    profile,
                    **common_options,
                    min_observations=settings.adaptive_min_observations,
                    latency_ewma_weight=(settings.adaptive_latency_ewma_weight),
                )
            self._vspec_adaptive_target_only = _TargetOnlyLatch()
            self._vspec_adaptive_summary_logged = False

        def adaptive_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
            active_request_ids = set(self.requests)
            target_only_latched = (
                settings.adaptive_gamma0_mode == "sticky"
                and self._vspec_adaptive_target_only.refresh(active_request_ids)
            )
            previous_max_num_running_reqs = self.max_num_running_reqs
            refill_batch = settings.adaptive_refill_batch
            if refill_batch == 0:
                refill_batch = 8 if settings.method == "draft_model" else 4
            num_running = len(self.running) + self.num_waiting_for_streaming_input
            num_waiting = len(self.waiting) + len(self.skipped_waiting)
            free_slots = previous_max_num_running_reqs - num_running
            refill_target = min(refill_batch, num_waiting)
            hold_waiting = (
                refill_batch > 1
                and num_running > 0
                and num_waiting > 0
                and free_slots < refill_target
            )
            isolate_refill = (
                os.environ.get(
                    "HUST_VSPEC_ADAPTIVE_ISOLATE_REFILL",
                    "0",
                )
                == "1"
                and refill_batch > 1
                and num_running > 0
                and num_waiting > 0
                and free_slots >= refill_target
            )
            if hold_waiting:
                self.max_num_running_reqs = num_running
            previous_running = None
            if isolate_refill:
                previous_running = self.running
                self.running = []
                self.max_num_running_reqs = free_slots
            try:
                scheduler_output = original_schedule(self, *args, **kwargs)
            finally:
                if previous_running is not None:
                    self.running = previous_running + self.running
                self.max_num_running_reqs = previous_max_num_running_reqs
            if not scheduler_output.num_scheduled_tokens:
                return scheduler_output

            batch_size = _adaptive_scheduled_batch_size(
                self,
                len(scheduler_output.num_scheduled_tokens),
            )
            context_tokens = (
                0
                if settings.adaptive_policy == "online"
                else _context_tokens_after_schedule(self, scheduler_output)
            )
            if target_only_latched:
                decision = self._vspec_adaptive_controller.force(
                    0,
                    batch_size,
                    context_tokens,
                    reason="target_only_latched",
                )
            else:
                decision = self._vspec_adaptive_controller.choose(
                    batch_size,
                    context_tokens,
                )
                if decision.gamma == 0 and settings.adaptive_gamma0_mode == "sticky":
                    self._vspec_adaptive_target_only.enter(set(self.requests))
            scheduler_output.num_spec_tokens_to_schedule = decision.gamma
            scheduler_output.vspec_adaptive_batch_size = batch_size
            scheduler_output.vspec_adaptive_context_tokens = context_tokens
            _update_async_next_frame_gamma(
                self,
                scheduler_output,
                decision.gamma,
            )
            self._vspec_adaptive_last_decision = decision
            if settings.adaptive_trace:
                position_acceptance = tuple(
                    round(rate, 4)
                    for rate in self._vspec_adaptive_controller.position_acceptance_rates
                )
                logger.warning(
                    "vSpec Adaptive decision: batch=%d context_tokens=%d "
                    "gamma=%d previous_gamma=%d acceptance=%.4f "
                    "position_acceptance=%s goodput=%.6f "
                    "previous_goodput=%.6f latency_factor=%.4f "
                    "latency_samples=%d gamma0_mode=%s reason=%s",
                    batch_size,
                    context_tokens,
                    decision.gamma,
                    decision.previous_gamma,
                    decision.acceptance_rate,
                    position_acceptance,
                    decision.predicted_goodput,
                    decision.previous_goodput,
                    self._vspec_adaptive_controller.latency_correction_factor(decision.gamma),
                    self._vspec_adaptive_controller.latency_observations[decision.gamma],
                    settings.adaptive_gamma0_mode,
                    decision.reason,
                )
            return scheduler_output

        def adaptive_update_from_output(
            self: Any,
            scheduler_output: Any,
            model_runner_output: Any,
        ) -> Any:
            controller = self._vspec_adaptive_controller
            if settings.adaptive_policy == "online":
                controller.begin_step_feedback()
            try:
                result = original_update_from_output(
                    self,
                    scheduler_output,
                    model_runner_output,
                )
            except Exception:
                if settings.adaptive_policy == "online":
                    controller.discard_step_feedback(
                        completed_at_ms=time.perf_counter() * 1000.0,
                    )
                raise
            if settings.adaptive_policy == "online":
                gamma = int(scheduler_output.num_spec_tokens_to_schedule)
                stable_decode = (
                    not bool(scheduler_output.scheduled_new_reqs)
                    and _runtime_target_query_width(scheduler_output) == gamma + 1
                )
                completed_at_ms = time.perf_counter() * 1000.0
                if stable_decode:
                    controller.complete_step(
                        gamma=gamma,
                        batch_size=int(scheduler_output.vspec_adaptive_batch_size),
                        context_tokens=int(scheduler_output.vspec_adaptive_context_tokens),
                        completed_at_ms=completed_at_ms,
                    )
                else:
                    controller.discard_step_feedback(
                        completed_at_ms=completed_at_ms,
                    )
            elif measure_step_latency and (
                measurement := getattr(
                    model_runner_output,
                    "vspec_adaptive_step_measurement",
                    None,
                )
            ):
                controller.observe_step_latency(
                    gamma=int(measurement["gamma"]),
                    batch_size=int(measurement["batch_size"]),
                    context_tokens=int(measurement["context_tokens"]),
                    latency_ms=float(measurement["latency_ms"]),
                )
            if (
                settings.adaptive_policy == "online"
                and not self.requests
                and not self._vspec_adaptive_summary_logged
            ):
                logger.warning(
                    "vSpec Online Adaptive summary: %s",
                    controller.summary(),
                )
                self._vspec_adaptive_summary_logged = True
            return result

        def adaptive_make_stats(
            self: Any,
            spec_decoding_stats: Any,
            num_draft_tokens: int,
            num_accepted_tokens: int,
            num_invalid_spec_tokens: dict[str, int] | None,
            request_id: str,
        ) -> Any:
            valid_draft_tokens = num_draft_tokens
            if num_invalid_spec_tokens:
                valid_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
            valid_draft_tokens = max(0, valid_draft_tokens)
            valid_accepted_tokens = min(num_accepted_tokens, valid_draft_tokens)
            self._vspec_adaptive_controller.observe(
                valid_draft_tokens,
                valid_accepted_tokens,
                request_id=request_id,
            )
            return original_make_stats(
                self,
                spec_decoding_stats,
                num_draft_tokens,
                num_accepted_tokens,
                num_invalid_spec_tokens,
                request_id,
            )

        def adaptive_update_draft_token_ids(
            self: Any,
            draft_token_ids: Any,
        ) -> Any:
            if settings.adaptive_entropy_stop:
                _trim_placeholder_suffixes(draft_token_ids)
            return original_update_draft_token_ids(self, draft_token_ids)

        Scheduler.__init__ = scheduler_init
        Scheduler.schedule = adaptive_schedule
        Scheduler.make_spec_decoding_stats = adaptive_make_stats
        Scheduler.update_from_output = adaptive_update_from_output
        Scheduler.update_draft_token_ids = adaptive_update_draft_token_ids
        setattr(Scheduler, SCHEDULER_PATCH_MARKER, True)
        applied = True

    if not getattr(AscendSpecDecodeBaseProposer, PROPOSER_PATCH_MARKER, False):
        original_propose = AscendSpecDecodeBaseProposer._propose
        original_dummy_run = AscendSpecDecodeBaseProposer.dummy_run

        def adaptive_dummy_run(self: Any, *args: Any, **kwargs: Any) -> Any:
            if settings.adaptive_entropy_stop:
                self._vspec_entropy_topk = settings.adaptive_entropy_topk
                _install_draft_entropy_probe(self)
            if not settings.adaptive_full_graph:
                return original_dummy_run(self, *args, **kwargs)
            configured_gamma = int(
                getattr(
                    self,
                    "_vspec_adaptive_configured_gamma",
                    self.num_speculative_tokens,
                )
            )
            self._vspec_adaptive_configured_gamma = configured_gamma
            return _capture_all_adaptive_draft_graphs(
                self,
                original_dummy_run,
                configured_gamma,
                settings.adaptive_min_gamma,
                *args,
                **kwargs,
            )

        def adaptive_propose(self: Any, *args: Any, **kwargs: Any) -> Any:
            scheduler_output = kwargs.get("scheduler_output")
            requested_gamma = getattr(
                scheduler_output,
                "num_spec_tokens_to_schedule",
                settings.adaptive_max_gamma,
            )
            requested_gamma = int(requested_gamma)
            if settings.adaptive_entropy_stop:
                self._vspec_entropy_topk = settings.adaptive_entropy_topk
                _install_draft_entropy_probe(self)
                self._vspec_entropy_recent_values = []
            configured_gamma = int(
                getattr(
                    self,
                    "_vspec_adaptive_configured_gamma",
                    self.num_speculative_tokens,
                )
            )
            self._vspec_adaptive_configured_gamma = configured_gamma
            if not settings.adaptive_min_gamma <= requested_gamma <= configured_gamma:
                raise RuntimeError(
                    "vSpec Adaptive runtime gamma is outside the configured range: "
                    f"requested={requested_gamma}, range="
                    f"[{settings.adaptive_min_gamma}, {configured_gamma}]"
                )
            if requested_gamma == 0:
                runner = getattr(self, "runner", None)
                input_batch = getattr(runner, "input_batch", None)
                num_reqs = int(getattr(input_batch, "num_reqs", 0))
                next_token_ids = kwargs.get("next_token_ids")
                if next_token_ids is None:
                    raise RuntimeError(
                        "vSpec Adaptive target-only proposal requires next_token_ids"
                    )
                if settings.adaptive_gamma0_mode == "sync":
                    previous_runnable = self._runnable
                    if settings.adaptive_full_graph:
                        self._runnable = getattr(
                            self,
                            "_vspec_adaptive_full_graph_body",
                            previous_runnable,
                        )
                    target_query_width = _proposal_input_query_width(
                        kwargs.get("common_attn_metadata"),
                        kwargs.get("target_model_batch_desc"),
                    )
                    try:
                        with _runtime_runner_query_width(
                            self,
                            target_query_width,
                        ):
                            with _runtime_eager_proposer(self, True):
                                with _runtime_eager_dispatch(self, True):
                                    synced = _run_with_runtime_gamma(
                                        self,
                                        1,
                                        original_propose,
                                        *args,
                                        **kwargs,
                                    )
                    finally:
                        self._runnable = previous_runnable
                    self._vspec_adaptive_previous_proposal_gamma = 0
                    return synced.new_empty((num_reqs, 0))
                self._vspec_adaptive_previous_proposal_gamma = 0
                return next_token_ids.new_empty((num_reqs, 0))

            previous_runnable = self._runnable
            previous_entropy_enabled = getattr(
                self,
                "_vspec_entropy_measure_enabled",
                True,
            )
            self._vspec_entropy_measure_enabled = requested_gamma >= _MIN_CONFIDENCE_STOP_GAMMA
            from vllm_ascend.compilation import acl_graph as acl_graph_module

            previous_graph_params = acl_graph_module._draft_graph_params
            previous_proposal_gamma = int(
                getattr(
                    self,
                    "_vspec_adaptive_previous_proposal_gamma",
                    requested_gamma,
                )
            )
            execution_gamma = _proposal_execution_gamma(
                getattr(self, "method", None),
                requested_gamma,
                previous_proposal_gamma,
            )
            graph_params_by_gamma = getattr(
                self,
                "_vspec_adaptive_graph_params_by_gamma",
                None,
            )
            if graph_params_by_gamma is not None:
                acl_graph_module._draft_graph_params = graph_params_by_gamma[execution_gamma]
            proposal_force_eager = False
            if settings.adaptive_full_graph:
                if previous_proposal_gamma == execution_gamma:
                    graph_batch_matches = _full_graph_batch_matches_capture(
                        kwargs.get("common_attn_metadata"),
                        kwargs.get("target_model_batch_desc"),
                        execution_gamma,
                        list(self.vllm_config.compilation_config.cudagraph_capture_sizes),
                    )
                    if graph_batch_matches:
                        self._runnable = _get_full_graph_runnable_for_gamma(
                            self,
                            execution_gamma,
                            configured_gamma,
                            settings.adaptive_min_gamma,
                        )
                    else:
                        proposal_force_eager = True
                        self._runnable = getattr(
                            self,
                            "_vspec_adaptive_full_graph_body",
                            previous_runnable,
                        )
                        if settings.adaptive_trace:
                            common_attn_metadata = kwargs.get("common_attn_metadata")
                            target_model_batch_desc = kwargs.get("target_model_batch_desc")
                            logger.warning(
                                "vSpec Adaptive Draft graph batch mismatch; "
                                "using eager for gamma=%d target_tokens=%s "
                                "target_reqs=%s metadata_batch=%s "
                                "metadata_actual_tokens=%s",
                                execution_gamma,
                                getattr(
                                    target_model_batch_desc,
                                    "num_tokens",
                                    None,
                                ),
                                getattr(
                                    target_model_batch_desc,
                                    "num_reqs",
                                    None,
                                ),
                                (
                                    common_attn_metadata.batch_size()
                                    if common_attn_metadata is not None
                                    else None
                                ),
                                getattr(
                                    common_attn_metadata,
                                    "num_actual_tokens",
                                    None,
                                ),
                            )
                else:
                    # The first pass consumes Target outputs produced with the
                    # previous gamma. Its static FULL-graph batch constants do
                    # not match the new gamma until the next scheduler step.
                    proposal_force_eager = True
                    self._runnable = getattr(
                        self,
                        "_vspec_adaptive_full_graph_body",
                        previous_runnable,
                    )
                    if settings.adaptive_trace:
                        logger.warning(
                            "vSpec Adaptive Draft transition uses eager: "
                            "previous_gamma=%d gamma=%d",
                            previous_proposal_gamma,
                            execution_gamma,
                        )
            try:
                proposal_kwargs = kwargs
                if proposal_force_eager:
                    proposal_kwargs = dict(kwargs)
                    proposal_kwargs["target_model_batch_desc"] = _nonuniform_batch_descriptor(
                        kwargs.get("target_model_batch_desc")
                    )
                target_query_width = _proposal_input_query_width(
                    kwargs.get("common_attn_metadata"),
                    kwargs.get("target_model_batch_desc"),
                )
                with _runtime_runner_query_width(
                    self,
                    target_query_width,
                ):
                    with _runtime_eager_proposer(
                        self,
                        proposal_force_eager,
                    ):
                        with _runtime_eager_dispatch(
                            self,
                            proposal_force_eager,
                        ):
                            result = _run_with_runtime_gamma(
                                self,
                                execution_gamma,
                                original_propose,
                                *args,
                                **proposal_kwargs,
                            )
                if execution_gamma != requested_gamma:
                    result = result[:, :requested_gamma]
                self._vspec_adaptive_previous_proposal_gamma = requested_gamma
                if draft_stopper is not None:
                    common_attn_metadata = kwargs.get("common_attn_metadata")
                    batch_size = (
                        int(common_attn_metadata.batch_size())
                        if common_attn_metadata is not None
                        else int(result.shape[0])
                    )
                    batch_size = min(batch_size, int(result.shape[0]))
                    entropies = _draft_entropy_matrix(
                        self,
                        requested_gamma,
                        batch_size,
                        settings.adaptive_entropy_topk,
                    )
                    if entropies is not None:
                        masked, lengths = draft_stopper.mask_draft_tokens(
                            result[:batch_size, :requested_gamma],
                            entropies,
                        )
                        if batch_size == int(result.shape[0]):
                            result = masked
                        else:
                            result = result.clone()
                            result[:batch_size, :requested_gamma] = masked
                        self._vspec_adaptive_last_draft_lengths = lengths
                        if settings.adaptive_trace:
                            import torch

                            counts = (
                                torch.bincount(
                                    lengths.to(torch.int64),
                                    minlength=requested_gamma + 1,
                                )
                                .cpu()
                                .tolist()
                            )
                            logger.warning(
                                "vSpec Adaptive current-round Draft lengths: budget=%d counts=%s",
                                requested_gamma,
                                {
                                    length: int(count)
                                    for length, count in enumerate(counts)
                                    if count
                                },
                            )
                    elif settings.adaptive_trace:
                        logger.warning(
                            "vSpec Adaptive current-round confidence data is "
                            "unavailable for gamma=%d; keeping the full proposal",
                            requested_gamma,
                        )
                return result
            finally:
                self._vspec_entropy_measure_enabled = previous_entropy_enabled
                self._runnable = previous_runnable
                acl_graph_module._draft_graph_params = previous_graph_params

        AscendSpecDecodeBaseProposer.dummy_run = adaptive_dummy_run
        AscendSpecDecodeBaseProposer._propose = adaptive_propose
        setattr(AscendSpecDecodeBaseProposer, PROPOSER_PATCH_MARKER, True)
        applied = True

        if AscendDflashProposer is not None and "dummy_run" in AscendDflashProposer.__dict__:
            original_dflash_dummy_run = AscendDflashProposer.dummy_run

            def adaptive_dflash_dummy_run(
                self: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                if not settings.adaptive_full_graph:
                    return original_dflash_dummy_run(
                        self,
                        *args,
                        **kwargs,
                    )
                configured_gamma = int(
                    getattr(
                        self,
                        "_vspec_adaptive_configured_gamma",
                        self.num_speculative_tokens,
                    )
                )
                self._vspec_adaptive_configured_gamma = configured_gamma
                return _capture_all_adaptive_draft_graphs(
                    self,
                    original_dflash_dummy_run,
                    configured_gamma,
                    settings.adaptive_min_gamma,
                    *args,
                    **kwargs,
                )

            AscendDflashProposer.dummy_run = adaptive_dflash_dummy_run

    if not getattr(NPUModelRunner, RUNNER_PATCH_MARKER, False):
        import torch

        original_runner_init = NPUModelRunner.__init__
        original_prepare_inputs = NPUModelRunner._prepare_inputs
        original_execute_model = NPUModelRunner.execute_model
        original_sample_tokens = NPUModelRunner.sample_tokens
        original_warmup_and_capture = NPUModelRunner._warmup_and_capture
        original_determine_batch_execution = NPUModelRunner._determine_batch_execution_and_padding
        original_copy_draft_tokens = NPUModelRunner._copy_draft_token_ids_to_cpu
        original_propose_draft_tokens = NPUModelRunner.propose_draft_token_ids
        original_profile_cudagraph_memory = NPUModelRunner.profile_cudagraph_memory
        original_capture_model = NPUModelRunner.capture_model
        original_check_cudagraph_mode = NPUModelRunner._check_and_update_cudagraph_mode

        def runner_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_runner_init(self, *args, **kwargs)
            existing_widths = getattr(
                self,
                "_nanoparl_uniform_decode_query_lens",
                (self.uniform_decode_query_len,),
            )
            self._nanoparl_uniform_decode_query_lens = _adaptive_target_query_lens(
                existing_widths,
                settings.adaptive_max_gamma,
            )
            if measure_step_latency:
                self._vspec_adaptive_pending_measurements = deque()

        def prepare_inputs(
            self: Any,
            scheduler_output: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            runtime_gamma = int(
                getattr(
                    scheduler_output,
                    "num_spec_tokens_to_schedule",
                    settings.adaptive_max_gamma,
                )
            )
            current_query_width = _runtime_target_query_width(scheduler_output)
            stable_decode = (
                not bool(scheduler_output.scheduled_new_reqs)
                and current_query_width == runtime_gamma + 1
            )
            with _runtime_dynamic_eagle_state_kernel(
                self,
                stable_decode=stable_decode,
                gamma=runtime_gamma,
            ):
                return original_prepare_inputs(
                    self,
                    scheduler_output,
                    *args,
                    **kwargs,
                )

        def execute_model(
            self: Any,
            scheduler_output: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if (
                settings.method == "eagle"
                and os.environ.get(_EAGLE_DYNAMIC_UNIFORM_STATE_ENV) == "1"
                and not hasattr(
                    self,
                    "_vspec_adaptive_configured_target_gamma",
                )
            ):
                configured_target_gamma = self.num_spec_tokens
                self._vspec_adaptive_configured_target_gamma = configured_target_gamma
                self.num_spec_tokens = (
                    settings.adaptive_min_gamma + settings.adaptive_max_gamma + 1
                ) // 2
                logger.warning(
                    "vSpec Adaptive EAGLE Target state width isolated: "
                    "configured=%d state_anchor=%d",
                    configured_target_gamma,
                    self.num_spec_tokens,
                )
            started_at = time.perf_counter() if measure_step_latency else None
            runtime_gamma = int(
                getattr(
                    scheduler_output,
                    "num_spec_tokens_to_schedule",
                    settings.adaptive_max_gamma,
                )
            )
            current_query_width = _runtime_target_query_width(scheduler_output)
            stable_decode = (
                not bool(scheduler_output.scheduled_new_reqs)
                and current_query_width == runtime_gamma + 1
            )
            previous_target_only = getattr(
                self,
                "_vspec_adaptive_target_only_step",
                False,
            )
            # With async scheduling, ``runtime_gamma`` controls the proposal
            # produced after this Target pass. The current pass still verifies
            # the previous proposal, whose width is represented by the actual
            # query layout. In particular, the 0 -> 1 recovery frame is a q1
            # Target pass and must not replay a wider combined FULL graph.
            self._vspec_adaptive_target_only_step = _current_target_is_target_only(
                current_query_width,
                runtime_gamma,
            )
            from vllm_ascend.compilation import acl_graph as acl_graph_module

            base_graph_params = acl_graph_module._graph_params
            # FULL_DECODE_ONLY has independent uniform decode descriptors for
            # each query width. Combined FULL shares one mixed descriptor and
            # event family; swapping its global graph-parameter table can
            # deadlock a repeated replay on Ascend.
            isolate_target_graph_params = _uses_width_isolated_target_graph_params(
                self.compilation_config.cudagraph_mode
            )
            graph_params_by_width = (
                _prepare_target_graph_params(
                    self,
                    base_graph_params,
                    self._nanoparl_uniform_decode_query_lens,
                )
                if isolate_target_graph_params
                else None
            )
            runtime_query_width = current_query_width
            if graph_params_by_width is not None and runtime_query_width in graph_params_by_width:
                acl_graph_module._graph_params = graph_params_by_width[runtime_query_width]
            rejection_sampler = getattr(self, "rejection_sampler", None)
            previous_confidence_accept = None
            had_confidence_accept_attr = False
            if settings.method == "eagle" and rejection_sampler is not None:
                had_confidence_accept_attr = hasattr(
                    rejection_sampler,
                    _EAGLE_CONFIDENCE_ACCEPT_ATTR,
                )
                previous_confidence_accept = getattr(
                    rejection_sampler,
                    _EAGLE_CONFIDENCE_ACCEPT_ATTR,
                    None,
                )
                setattr(
                    rejection_sampler,
                    _EAGLE_CONFIDENCE_ACCEPT_ATTR,
                    _current_eagle_confidence_accept_enabled(
                        current_query_width,
                        runtime_gamma,
                    ),
                )
            try:
                if settings.adaptive_trace:
                    logger.warning(
                        "vSpec Adaptive Target execute start: gamma=%d "
                        "query_width=%s stable_decode=%s",
                        runtime_gamma,
                        current_query_width,
                        stable_decode,
                    )
                result = original_execute_model(
                    self,
                    scheduler_output,
                    *args,
                    **kwargs,
                )
                if settings.adaptive_trace:
                    logger.warning(
                        "vSpec Adaptive Target execute complete: gamma=%d result_type=%s",
                        runtime_gamma,
                        type(result).__name__,
                    )
                if not measure_step_latency:
                    return result

                assert started_at is not None
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                output = _unwrap_model_runner_output(result)
                measurement = None
                if stable_decode:
                    measurement = {
                        "gamma": runtime_gamma,
                        "batch_size": int(
                            getattr(
                                scheduler_output,
                                "vspec_adaptive_batch_size",
                                len(scheduler_output.num_scheduled_tokens),
                            )
                        ),
                        "context_tokens": int(
                            getattr(
                                scheduler_output,
                                "vspec_adaptive_context_tokens",
                                0,
                            )
                        ),
                        "started_at": started_at,
                    }
                if hasattr(output, "req_ids"):
                    if measurement is not None:
                        output.vspec_adaptive_step_measurement = measurement
                        if output is result:
                            measurement["latency_ms"] = float(
                                getattr(
                                    output,
                                    "execution_time_ms",
                                    elapsed_ms,
                                )
                            )
                            measurement.pop("started_at", None)
                elif result is None:
                    self._vspec_adaptive_pending_measurements.append(measurement)
                if (
                    settings.adaptive_trace
                    and measurement is not None
                    and hasattr(output, "req_ids")
                    and "latency_ms" in measurement
                ):
                    measured_ms = measurement["latency_ms"]
                    logger.warning(
                        "vSpec Adaptive latency staged: gamma=%d "
                        "query_width=%d batch=%d latency_ms=%.4f",
                        runtime_gamma,
                        current_query_width,
                        measurement["batch_size"],
                        measured_ms,
                    )
                return result
            finally:
                if settings.method == "eagle" and rejection_sampler is not None:
                    if had_confidence_accept_attr:
                        setattr(
                            rejection_sampler,
                            _EAGLE_CONFIDENCE_ACCEPT_ATTR,
                            previous_confidence_accept,
                        )
                    else:
                        delattr(
                            rejection_sampler,
                            _EAGLE_CONFIDENCE_ACCEPT_ATTR,
                        )
                acl_graph_module._graph_params = base_graph_params
                self._vspec_adaptive_target_only_step = previous_target_only

        def sample_tokens(
            self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if settings.adaptive_trace:
                logger.warning("vSpec Adaptive Target sample start")
            result = original_sample_tokens(self, *args, **kwargs)
            if settings.adaptive_trace:
                logger.warning(
                    "vSpec Adaptive Target sample complete: result_type=%s",
                    type(result).__name__,
                )
            if not measure_step_latency:
                return result
            pending = self._vspec_adaptive_pending_measurements
            measurement = pending.popleft() if pending else None
            if measurement is None:
                return result
            output = _unwrap_model_runner_output(result)
            if not hasattr(output, "req_ids"):
                raise RuntimeError(
                    "vSpec Adaptive cannot attach latency to sampler output "
                    f"type {type(output).__name__}"
                )
            started_at = float(measurement.pop("started_at"))
            if output is result:
                measurement["latency_ms"] = (time.perf_counter() - started_at) * 1000.0
            else:
                measurement["started_at"] = started_at
            output.vspec_adaptive_step_measurement = measurement
            if settings.adaptive_trace and "latency_ms" in measurement:
                logger.warning(
                    "vSpec Adaptive latency staged after sampling: "
                    "gamma=%d batch=%d latency_ms=%.4f",
                    measurement["gamma"],
                    measurement["batch_size"],
                    measurement["latency_ms"],
                )
            return result

        def warmup_and_capture(
            self: Any,
            desc: Any,
            cudagraph_runtime_mode: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if (
                not settings.adaptive_full_graph
                or cudagraph_runtime_mode != CUDAGraphMode.FULL
                or not desc.uniform
                or not _uses_width_isolated_target_graph_params(
                    self.compilation_config.cudagraph_mode
                )
            ):
                return original_warmup_and_capture(
                    self,
                    desc,
                    cudagraph_runtime_mode,
                    *args,
                    **kwargs,
                )

            if desc.num_reqs is None or desc.num_reqs <= 0:
                raise RuntimeError("adaptive Target graph requires a positive request count")
            if desc.num_tokens % desc.num_reqs:
                raise RuntimeError("adaptive Target graph query width is not integral")
            query_width = desc.num_tokens // desc.num_reqs
            from vllm_ascend.compilation import acl_graph as acl_graph_module

            base_graph_params = acl_graph_module._graph_params
            graph_params_by_width = _prepare_target_graph_params(
                self,
                base_graph_params,
                self._nanoparl_uniform_decode_query_lens,
            )
            if graph_params_by_width is None or query_width not in graph_params_by_width:
                raise RuntimeError(
                    f"missing adaptive Target graph-parameter table for query width {query_width}"
                )
            acl_graph_module._graph_params = graph_params_by_width[query_width]
            try:
                with _runner_query_width(self, query_width):
                    return original_warmup_and_capture(
                        self,
                        desc,
                        cudagraph_runtime_mode,
                        *args,
                        **kwargs,
                    )
            finally:
                acl_graph_module._graph_params = base_graph_params

        def determine_batch_execution_and_padding(
            self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Combined FULL graphs have one mixed descriptor family. A q1
            # target-only step can collide with a graph captured at q(max+1),
            # whose static FIA metadata is incompatible. FULL_DECODE_ONLY has
            # width-specific uniform descriptors and remains graph-backed.
            if (
                getattr(self, "_vspec_adaptive_target_only_step", False)
                and self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
            ):
                if len(args) >= 7:
                    mutable_args = list(args)
                    mutable_args[6] = True
                    args = tuple(mutable_args)
                else:
                    kwargs["force_eager"] = True
            return original_determine_batch_execution(
                self,
                *args,
                **kwargs,
            )

        def prepare_adaptive_full_graphs(self: Any) -> None:
            drafter = getattr(self, "drafter", None)
            if drafter is None or not settings.adaptive_full_graph:
                return
            configured_gamma = int(
                getattr(
                    drafter,
                    "_vspec_adaptive_configured_gamma",
                    drafter.num_speculative_tokens,
                )
            )
            drafter._vspec_adaptive_configured_gamma = configured_gamma
            _initialize_full_graph_runnables(
                drafter,
                configured_gamma,
                settings.adaptive_min_gamma,
            )

        def propose_draft_token_ids(
            self: Any,
            valid_sampled_token_ids: Any,
            sampling_metadata: Any,
            scheduler_output: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            requested_gamma = int(
                getattr(
                    scheduler_output,
                    "num_spec_tokens_to_schedule",
                    settings.adaptive_max_gamma,
                )
            )
            if (
                requested_gamma != 0
                or self.use_async_scheduling
                or settings.adaptive_gamma0_mode == "sync"
            ):
                # Async scheduling still needs the generic next-token state
                # preparation. adaptive_propose() returns before Draft forward
                # once that state has been staged.
                return original_propose_draft_tokens(
                    self,
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    *args,
                    **kwargs,
                )

            drafter = getattr(self, "drafter", None)
            if drafter is not None:
                drafter._vspec_adaptive_previous_proposal_gamma = 0
            num_reqs = int(self.input_batch.num_reqs)
            if torch.is_tensor(valid_sampled_token_ids):
                return valid_sampled_token_ids.new_empty((num_reqs, 0))
            return torch.empty(
                (num_reqs, 0),
                dtype=torch.int64,
                device=self.device,
            )

        def profile_cudagraph_memory(self: Any, *args: Any, **kwargs: Any) -> Any:
            prepare_adaptive_full_graphs(self)
            return original_profile_cudagraph_memory(self, *args, **kwargs)

        def capture_model(self: Any, *args: Any, **kwargs: Any) -> Any:
            prepare_adaptive_full_graphs(self)
            return original_capture_model(self, *args, **kwargs)

        def copy_draft_tokens_to_cpu(
            self: Any,
            scheduler_output: Any,
            zeros_only: bool = False,
        ) -> None:
            draft_token_ids = self._draft_token_ids
            destination = self.draft_token_ids_cpu
            if (
                not torch.is_tensor(draft_token_ids)
                or draft_token_ids.ndim < 2
                or destination is None
                or draft_token_ids.shape[1] == destination.shape[1]
            ):
                return original_copy_draft_tokens(
                    self,
                    scheduler_output,
                    zeros_only,
                )

            self.prev_num_spec_tokens = draft_token_ids.shape[1]
            custom_class_enabled = getattr(
                self,
                "_nanoparl_async_custom_class_enabled",
                None,
            )
            if callable(custom_class_enabled) and custom_class_enabled():
                self._draft_token_req_ids = self.input_batch.req_ids.copy()
                return
            if self.use_async_scheduling and not (
                scheduler_output.has_structured_output_requests
                or self.input_batch.sampling_metadata.output_token_ids
            ):
                return
            self._draft_token_req_ids = self.input_batch.req_ids.copy()
            assert self.draft_token_ids_event is not None
            assert self.draft_token_ids_copy_stream is not None
            default_stream = torch.npu.current_stream()
            with torch.npu.stream(self.draft_token_ids_copy_stream):
                if not zeros_only:
                    self.draft_token_ids_copy_stream.wait_stream(default_stream)
                _copy_dynamic_draft_tokens(
                    destination,
                    draft_token_ids,
                    zeros_only=zeros_only,
                )
                self.draft_token_ids_event.record()

        def check_and_update_cudagraph_mode(
            self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not settings.adaptive_full_graph or original_restore_capture_sizes is not None:
                return original_check_cudagraph_mode(self, *args, **kwargs)

            requested_capture_sizes = list(self.compilation_config.cudagraph_capture_sizes or ())
            dispatcher = self.cudagraph_dispatcher
            original_initialize = dispatcher.initialize_cudagraph_keys

            def initialize_cudagraph_keys(
                cudagraph_mode: Any,
                uniform_decode_query_len: int = 1,
            ) -> Any:
                restore_adaptive_capture_sizes(
                    self.compilation_config,
                    method=(
                        self.speculative_config.method
                        if self.speculative_config is not None
                        else None
                    ),
                    requested_capture_sizes=requested_capture_sizes,
                )
                if getattr(cudagraph_mode, "name", None) == "FULL_DECODE_ONLY":
                    _initialize_adaptive_decode_only_graph_keys(
                        dispatcher,
                        cudagraph_mode,
                        self._nanoparl_uniform_decode_query_lens,
                    )
                    return None
                result = original_initialize(cudagraph_mode, uniform_decode_query_len)
                _add_adaptive_decode_graph_keys(
                    dispatcher,
                    self._nanoparl_uniform_decode_query_lens,
                )
                return result

            dispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys
            try:
                return original_check_cudagraph_mode(self, *args, **kwargs)
            finally:
                dispatcher.initialize_cudagraph_keys = original_initialize

        NPUModelRunner.__init__ = runner_init
        NPUModelRunner._prepare_inputs = prepare_inputs
        NPUModelRunner.execute_model = execute_model
        NPUModelRunner.sample_tokens = sample_tokens
        NPUModelRunner._warmup_and_capture = warmup_and_capture
        NPUModelRunner._determine_batch_execution_and_padding = (
            determine_batch_execution_and_padding
        )
        NPUModelRunner.profile_cudagraph_memory = profile_cudagraph_memory
        NPUModelRunner.capture_model = capture_model
        NPUModelRunner.propose_draft_token_ids = propose_draft_token_ids
        NPUModelRunner._copy_draft_token_ids_to_cpu = copy_draft_tokens_to_cpu
        NPUModelRunner._check_and_update_cudagraph_mode = check_and_update_cudagraph_mode
        setattr(NPUModelRunner, RUNNER_PATCH_MARKER, True)
        applied = True

    return applied
