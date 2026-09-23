"""Parallel FIA task rebinding for serial Draft FULL graphs."""

from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

PATCH_MARKER = "_vllm_hust_vspec_draft_parallel_graph_update_patched"
DEFER_WAIT_PATCH_MARKER = (
    "_vllm_hust_vspec_deferred_graph_update_wait_patched"
)
_UPDATE_THREAD_STATE = threading.local()


class _ParallelUpdateState:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="vspec-draft-fia-update",
        )
        self.streams: list[Any] | None = None
        self.device: int | None = None
        self.lock = threading.Lock()
        self.pending: list[Any] = []
        self.plans: dict[
            tuple[str, int],
            tuple[tuple[_UpdateDescriptor, ...], ...],
        ] = {}
        self.pipeline_plans: dict[
            tuple[str, int, int],
            tuple[
                tuple[_UpdateDescriptor, ...],
                tuple[tuple[_UpdateDescriptor, ...], ...],
            ],
        ] = {}

    def ensure_streams(self, torch: Any) -> tuple[int, list[Any]]:
        device = int(torch.npu.current_device())
        with self.lock:
            if self.streams is None:
                self.device = device
                self.streams = [
                    torch.npu.Stream(device=device)
                    for _ in range(self.workers)
                ]
            elif self.device != device:
                raise RuntimeError(
                    "vSpec Draft FIA updater cannot move between NPU devices"
                )
            return device, self.streams

    def wait_pending(self) -> None:
        pending, self.pending = self.pending, []
        for future in pending:
            future.result()

class _DenseFIAParam(NamedTuple):
    query: Any
    key: Any
    value: Any
    attn_mask: Any
    block_size: Any
    num_kv_heads: Any
    num_heads: Any
    scale: Any
    output: Any
    softmax_lse: Any
    sparse_mode: Any
    pre_tokens: Any
    next_tokens: Any
    sliding_window: Any
    c8_k_scale: Any
    c8_v_scale: Any
    layer_name: str | None
    adaptive_decision: Any


class _UpdateDescriptor(NamedTuple):
    param: _DenseFIAParam
    handle: Any
    event: Any
    metadata_step: int
    metadata_key: str
    sparse_mode: Any


def _normalize_dense_fia_param(
    param: tuple[Any, ...],
) -> _DenseFIAParam | None:
    """Normalize the dense FIA capture tuples used by supported Ascend hosts."""
    if len(param) == 22 and not isinstance(param[20], str):
        # Current host ABI includes sliding_window before the C8 arguments.
        return _DenseFIAParam(
            query=param[0],
            key=param[1],
            value=param[2],
            attn_mask=param[4],
            block_size=param[5],
            num_kv_heads=param[8],
            num_heads=param[9],
            scale=param[10],
            output=param[11],
            softmax_lse=param[12],
            sparse_mode=param[13],
            pre_tokens=param[14],
            next_tokens=param[15],
            sliding_window=param[16],
            c8_k_scale=param[17],
            c8_v_scale=param[19],
            layer_name=param[21],
            adaptive_decision=None,
        )
    if len(param) in (21, 22) and isinstance(param[20], str):
        # Older host ABI has no sliding_window field in the capture tuple.
        return _DenseFIAParam(
            query=param[0],
            key=param[1],
            value=param[2],
            attn_mask=param[4],
            block_size=param[5],
            num_kv_heads=param[8],
            num_heads=param[9],
            scale=param[10],
            output=param[11],
            softmax_lse=param[12],
            sparse_mode=param[13],
            pre_tokens=param[14],
            next_tokens=param[15],
            sliding_window=None,
            c8_k_scale=param[16],
            c8_v_scale=param[18],
            layer_name=param[20],
            adaptive_decision=param[21] if len(param) == 22 else None,
        )
    return None


def _layer_index(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else 0


def _chunk_descriptors(
    descriptors: list[_UpdateDescriptor],
    workers: int,
) -> tuple[tuple[_UpdateDescriptor, ...], ...]:
    return tuple(
        tuple(descriptors[offset::workers]) for offset in range(workers)
    )


def _pipeline_descriptors(
    descriptors: list[_UpdateDescriptor],
    workers: int,
    prefix_size: int,
) -> tuple[
    tuple[_UpdateDescriptor, ...],
    tuple[tuple[_UpdateDescriptor, ...], ...],
]:
    """Keep early layers on the caller and split the tail contiguously."""
    prefix_size = min(max(prefix_size, 1), len(descriptors))
    prefix = tuple(descriptors[:prefix_size])
    tail = descriptors[prefix_size:]
    chunk_size = max(1, (len(tail) + workers - 1) // workers)
    chunks = tuple(
        tuple(tail[offset : offset + chunk_size])
        for offset in range(0, len(tail), chunk_size)
    )
    if len(chunks) < workers:
        chunks += ((),) * (workers - len(chunks))
    return prefix, chunks[:workers]


def _graph_plan_scope(
    role: str,
    graph_params: Any,
    captured: Any,
) -> str:
    """Keep cached update descriptors local to one captured graph table."""
    return f"{role}:{id(graph_params)}:{len(captured)}"


def apply_draft_parallel_graph_update_patch(
    workers: int,
    target_workers: int = 0,
) -> bool:
    """Submit independent Draft FIA task updates on multiple NPU streams."""
    if workers <= 1 and target_workers <= 1:
        return False

    import torch
    import torch_npu
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.compilation import acl_graph as acl_graph_module

    impl = AscendAttentionBackendImpl
    if getattr(impl, PATCH_MARKER, False):
        return False

    original_update = impl.update_graph_params
    state = _ParallelUpdateState(workers) if workers > 1 else None
    target_state = None
    if target_workers > 1:
        target_state = (
            state
            if state is not None and target_workers == workers
            else _ParallelUpdateState(target_workers)
        )

    def update_one_stream(
        device: int,
        stream: Any,
        gate: Any,
        descriptors: tuple[_UpdateDescriptor, ...],
        attn_metadata: Any,
        workspace: Any,
    ) -> None:
        if getattr(_UPDATE_THREAD_STATE, "device", None) != device:
            torch.npu.set_device(device)
            _UPDATE_THREAD_STATE.device = device
        stream.wait_event(gate)
        with torch.npu.stream(stream):
            for descriptor in descriptors:
                normalized = descriptor.param
                metadata = (
                    attn_metadata[descriptor.metadata_key]
                    if descriptor.metadata_step < 0
                    else attn_metadata[descriptor.metadata_step][
                        descriptor.metadata_key
                    ]
                )
                torch.npu.graph_task_update_begin(
                    stream,
                    descriptor.handle,
                )
                torch_npu.npu_fused_infer_attention_score.out(
                    query=normalized.query,
                    key=normalized.key,
                    value=normalized.value,
                    block_table=metadata.block_tables,
                    atten_mask=normalized.attn_mask,
                    input_layout="TND",
                    block_size=normalized.block_size,
                    actual_seq_lengths=metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=metadata.seq_lens_list,
                    num_key_value_heads=normalized.num_kv_heads,
                    num_heads=normalized.num_heads,
                    scale=normalized.scale,
                    sparse_mode=descriptor.sparse_mode,
                    pre_tokens=normalized.pre_tokens,
                    next_tokens=normalized.next_tokens,
                    workspace=workspace,
                    out=[normalized.output, normalized.softmax_lse],
                )
                torch.npu.graph_task_update_end(stream)
                descriptor.event.record(stream)

    def dispatch_updates(
        update_state: _ParallelUpdateState,
        update_stream: Any,
        chunks: tuple[tuple[_UpdateDescriptor, ...], ...],
        attn_metadata: Any,
        workspace: Any,
        *,
        defer: bool,
    ) -> None:
        device, streams = update_state.ensure_streams(torch)
        gate = torch.npu.Event()
        gate.record(update_stream)
        if defer:
            update_state.pending = [
                update_state.executor.submit(
                    update_one_stream,
                    device,
                    stream,
                    gate,
                    chunk,
                    attn_metadata,
                    workspace,
                )
                for stream, chunk in zip(streams, chunks, strict=True)
                if chunk
            ]
            return

        futures = [
            update_state.executor.submit(
                update_one_stream,
                device,
                stream,
                gate,
                chunk,
                attn_metadata,
                workspace,
            )
            for stream, chunk in zip(streams[1:], chunks[1:], strict=True)
            if chunk
        ]
        error: BaseException | None = None
        try:
            if chunks[0]:
                update_one_stream(
                    device,
                    streams[0],
                    gate,
                    chunks[0],
                    attn_metadata,
                    workspace,
                )
        except BaseException as exc:
            error = exc
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def dispatch_pipelined_updates(
        update_state: _ParallelUpdateState,
        update_stream: Any,
        prefix: tuple[_UpdateDescriptor, ...],
        chunks: tuple[tuple[_UpdateDescriptor, ...], ...],
        attn_metadata: Any,
        workspace: Any,
    ) -> None:
        """Release early graph layers, then finish later bindings in background."""
        device, streams = update_state.ensure_streams(torch)
        gate = torch.npu.Event()
        gate.record(update_stream)
        pending = [
            update_state.executor.submit(
                update_one_stream,
                device,
                stream,
                gate,
                chunk,
                attn_metadata,
                workspace,
            )
            for stream, chunk in zip(streams, chunks, strict=True)
            if chunk
        ]
        try:
            update_one_stream(
                device,
                update_stream,
                gate,
                prefix,
                attn_metadata,
                workspace,
            )
        except BaseException:
            for future in pending:
                future.result()
            raise
        update_state.pending = pending

    def update_graph_params(
        update_stream: Any,
        forward_context: Any,
        num_tokens: int,
        vllm_config: Any,
        speculative_config: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        draft_attn_metadatas = kwargs.get("draft_attn_metadatas")
        if draft_attn_metadatas is None and args:
            candidate = args[-1]
            if isinstance(candidate, list):
                draft_attn_metadatas = candidate

        if not _EXTRA_CTX.is_draft_model:
            if target_state is None or _EXTRA_CTX.sinks:
                return original_update(
                    update_stream,
                    forward_context,
                    num_tokens,
                    vllm_config,
                    speculative_config,
                    *args,
                    **kwargs,
                )
            target_state.wait_pending()
            graph_params = acl_graph_module.get_graph_params()
            captured = graph_params.attn_params.get(num_tokens, ())
            handles = graph_params.handles.get(num_tokens, ())
            events = graph_params.events.get(num_tokens, ())
            attn_metadata = forward_context.attn_metadata
            if (
                len(captured) < target_state.workers
                or len(captured) != len(handles)
                or len(captured) != len(events)
            ):
                return original_update(
                    update_stream,
                    forward_context,
                    num_tokens,
                    vllm_config,
                    speculative_config,
                    *args,
                    **kwargs,
                )

            pipeline_target = (
                os.environ.get(
                    "VSPEC_DRAFT_PIPELINED_TARGET_GRAPH_UPDATES"
                )
                == "1"
            )
            prefix_size = int(
                os.environ.get("VSPEC_DRAFT_TARGET_UPDATE_PREFIX", "4")
            )
            plan_name = _graph_plan_scope(
                "target",
                graph_params,
                captured,
            )
            plan_key = (plan_name, num_tokens)
            pipeline_key = (plan_name, num_tokens, prefix_size)
            pipeline_plan = (
                target_state.pipeline_plans.get(pipeline_key)
                if pipeline_target
                else None
            )
            chunks = (
                target_state.plans.get(plan_key)
                if not pipeline_target
                else None
            )
            if chunks is None and pipeline_plan is None:
                metadata_keys = sorted(
                    (
                        key
                        for key, metadata in attn_metadata.items()
                        if hasattr(metadata, "seq_lens_list")
                    ),
                    key=_layer_index,
                )
                if len(metadata_keys) < len(captured):
                    return original_update(
                        update_stream,
                        forward_context,
                        num_tokens,
                        vllm_config,
                        speculative_config,
                        *args,
                        **kwargs,
                    )
                target_descriptors: list[_UpdateDescriptor] = []
                for index, (param, handle, event) in enumerate(
                    zip(captured, handles, events, strict=True)
                ):
                    if not isinstance(param, tuple):
                        break
                    normalized = _normalize_dense_fia_param(param)
                    if normalized is None:
                        break
                    layer_name = normalized.layer_name
                    metadata_key = (
                        layer_name
                        if layer_name is not None
                        and layer_name in attn_metadata
                        else metadata_keys[index]
                    )
                    metadata = attn_metadata[metadata_key]
                    if (
                        normalized.sliding_window is not None
                        or normalized.c8_k_scale is not None
                        or normalized.c8_v_scale is not None
                        or normalized.adaptive_decision is not None
                        or getattr(metadata, "tree_attn", False)
                        or not metadata.causal
                    ):
                        break
                    target_descriptors.append(
                        _UpdateDescriptor(
                            normalized,
                            handle,
                            event,
                            -1,
                            metadata_key,
                            normalized.sparse_mode,
                        )
                    )
                if len(target_descriptors) != len(captured):
                    return original_update(
                        update_stream,
                        forward_context,
                        num_tokens,
                        vllm_config,
                        speculative_config,
                        *args,
                        **kwargs,
                    )
                if pipeline_target:
                    pipeline_plan = _pipeline_descriptors(
                        target_descriptors,
                        target_state.workers,
                        prefix_size,
                    )
                    target_state.pipeline_plans[pipeline_key] = pipeline_plan
                else:
                    chunks = _chunk_descriptors(
                        target_descriptors,
                        target_state.workers,
                    )
                    target_state.plans[plan_key] = chunks

            workspace = graph_params.workspaces.get(num_tokens)
            if pipeline_target:
                assert pipeline_plan is not None
                prefix, pipeline_chunks = pipeline_plan
                dispatch_pipelined_updates(
                    target_state,
                    update_stream,
                    prefix,
                    pipeline_chunks,
                    attn_metadata,
                    workspace,
                )
            else:
                assert chunks is not None
                dispatch_updates(
                    target_state,
                    update_stream,
                    chunks,
                    attn_metadata,
                    workspace,
                    defer=(
                        os.environ.get(
                            "VSPEC_DRAFT_DEFER_TARGET_GRAPH_UPDATES"
                        )
                        == "1"
                    ),
                )
            return None

        if state is None:
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                *args,
                **kwargs,
            )
        state.wait_pending()
        if _EXTRA_CTX.sinks or not draft_attn_metadatas:
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                *args,
                **kwargs,
            )

        graph_params = acl_graph_module.get_draft_graph_params()
        captured = graph_params.attn_params.get(num_tokens, ())
        handles = graph_params.handles.get(num_tokens, ())
        events = graph_params.events.get(num_tokens, ())
        if (
            len(captured) < workers
            or len(captured) != len(handles)
            or len(captured) != len(events)
        ):
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                *args,
                **kwargs,
            )

        plan_name = (
            f"{_graph_plan_scope('draft', graph_params, captured)}:"
            f"{len(draft_attn_metadatas)}"
        )
        plan_key = (plan_name, num_tokens)
        pipeline_draft = (
            os.environ.get("VSPEC_DRAFT_PIPELINED_GRAPH_UPDATES") == "1"
        )
        prefix_size = int(
            os.environ.get("VSPEC_DRAFT_GRAPH_UPDATE_PREFIX", "2")
        )
        pipeline_key = (plan_name, num_tokens, prefix_size)
        pipeline_plan = (
            state.pipeline_plans.get(pipeline_key)
            if pipeline_draft
            else None
        )
        chunks = state.plans.get(plan_key) if not pipeline_draft else None
        if chunks is None and pipeline_plan is None:
            draft_steps = [
                (step, key)
                for step, metadata in enumerate(draft_attn_metadatas)
                for key in metadata
            ]
            if not draft_steps:
                return original_update(
                    update_stream,
                    forward_context,
                    num_tokens,
                    vllm_config,
                    speculative_config,
                    *args,
                    **kwargs,
                )
            if len(captured) > len(draft_steps):
                repeats = (len(captured) + len(draft_steps) - 1) // len(
                    draft_steps
                )
                draft_steps = (draft_steps * repeats)[: len(captured)]
            else:
                draft_steps = draft_steps[: len(captured)]

            descriptors: list[_UpdateDescriptor] = []
            for param, handle, event, (step, key) in zip(
                captured,
                handles,
                events,
                draft_steps,
                strict=True,
            ):
                if not isinstance(param, tuple):
                    break
                normalized = _normalize_dense_fia_param(param)
                metadata = draft_attn_metadatas[step][key]
                if (
                    normalized is None
                    or normalized.sliding_window is not None
                    or normalized.c8_k_scale is not None
                    or normalized.c8_v_scale is not None
                    or not hasattr(metadata, "seq_lens_list")
                ):
                    break
                sparse_mode = (
                    normalized.sparse_mode if metadata.causal else 0
                )
                descriptors.append(
                    _UpdateDescriptor(
                        normalized,
                        handle,
                        event,
                        step,
                        key,
                        sparse_mode,
                    )
                )

            if len(descriptors) != len(captured):
                return original_update(
                    update_stream,
                    forward_context,
                    num_tokens,
                    vllm_config,
                    speculative_config,
                    *args,
                    **kwargs,
                )
            if pipeline_draft:
                pipeline_plan = _pipeline_descriptors(
                    descriptors,
                    workers,
                    prefix_size,
                )
                state.pipeline_plans[pipeline_key] = pipeline_plan
            else:
                chunks = _chunk_descriptors(descriptors, workers)
                state.plans[plan_key] = chunks

        workspace = graph_params.workspaces.get(num_tokens)
        if pipeline_draft:
            assert pipeline_plan is not None
            prefix, pipeline_chunks = pipeline_plan
            dispatch_pipelined_updates(
                state,
                update_stream,
                prefix,
                pipeline_chunks,
                draft_attn_metadatas,
                workspace,
            )
        else:
            assert chunks is not None
            dispatch_updates(
                state,
                update_stream,
                chunks,
                draft_attn_metadatas,
                workspace,
                defer=(
                    os.environ.get("VSPEC_DRAFT_DEFER_GRAPH_UPDATES")
                    == "1"
                ),
            )

    impl.update_graph_params = staticmethod(update_graph_params)
    setattr(impl, PATCH_MARKER, True)

    # A deferred task update prepares the *next* replay.  Waiting from the
    # following update_graph_params() call is too late because that replay has
    # already been launched.  Retire the host futures immediately before the
    # matching graph wrapper runs; work submitted after the previous replay can
    # then overlap sampling and execution of the other model.
    graph_wrapper = acl_graph_module.ACLGraphWrapper
    if not getattr(graph_wrapper, DEFER_WAIT_PATCH_MARKER, False):
        original_graph_call = graph_wrapper.__call__

        def graph_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            if _EXTRA_CTX.is_draft_model:
                if (
                    state is not None
                    and os.environ.get("VSPEC_DRAFT_DEFER_GRAPH_UPDATES")
                    == "1"
                ):
                    state.wait_pending()
            elif (
                target_state is not None
                and os.environ.get(
                    "VSPEC_DRAFT_DEFER_TARGET_GRAPH_UPDATES"
                )
                == "1"
            ):
                target_state.wait_pending()
            return original_graph_call(self, *args, **kwargs)

        graph_wrapper.__call__ = graph_call
        setattr(graph_wrapper, DEFER_WAIT_PATCH_MARKER, True)
    return True
