"""Parallel FIA task rebinding for dense EAGLE Target FULL graphs."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

PATCH_MARKER = "_vllm_hust_vspec_parallel_graph_update_patched"


class _ParallelUpdateState:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="vspec-fia-update",
        )
        self.streams: list[Any] | None = None
        self.device: int | None = None
        self.lock = threading.Lock()

    def ensure_streams(self, torch: Any) -> tuple[int, list[Any]]:
        device = int(torch.npu.current_device())
        with self.lock:
            if self.streams is None:
                self.device = device
                self.streams = [torch.npu.Stream(device=device) for _ in range(self.workers)]
            elif self.device != device:
                raise RuntimeError("vSpec parallel FIA updater cannot move between NPU devices")
            return device, self.streams


def _layer_index(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else 0


def apply_parallel_graph_update_patch(workers: int) -> bool:
    """Submit independent Target-layer FIA task updates on multiple streams."""
    if workers <= 1:
        return False

    import torch
    import torch_npu
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.compilation import acl_graph as acl_graph_module
    from vllm_ascend.compilation.acl_graph import _EXTRA_CTX

    impl = AscendAttentionBackendImpl
    if getattr(impl, PATCH_MARKER, False):
        return False

    original_update = impl.update_graph_params
    state = _ParallelUpdateState(workers)

    def update_one_stream(
        device: int,
        stream: Any,
        gate: Any,
        items: list[tuple[Any, ...]],
        workspace: Any,
    ) -> None:
        torch.npu.set_device(device)
        stream.wait_event(gate)
        with torch.npu.stream(stream):
            for item in items:
                (
                    param,
                    handle,
                    event,
                    seq_lens,
                    actual_seq_lengths_q,
                    block_tables,
                ) = item
                if len(param) == 21:
                    param = (*param, None)
                (
                    query,
                    key_cache,
                    value,
                    _captured_block_tables,
                    attn_mask,
                    block_size,
                    _captured_seq_lens,
                    _query_start_loc,
                    num_kv_heads,
                    num_heads,
                    scale,
                    attn_output,
                    softmax_lse,
                    sparse_mode,
                    pre_tokens,
                    next_tokens,
                    _c8_k_aq_scale,
                    _c8_k_aq_offset,
                    _c8_v_aq_scale,
                    _c8_v_aq_offset,
                    _layer_name,
                    _adaptive_graph_decision,
                ) = param
                torch.npu.graph_task_update_begin(stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(
                    query=query,
                    key=key_cache,
                    value=value,
                    block_table=block_tables,
                    atten_mask=attn_mask,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=actual_seq_lengths_q,
                    actual_seq_lengths_kv=seq_lens,
                    num_key_value_heads=num_kv_heads,
                    num_heads=num_heads,
                    scale=scale,
                    sparse_mode=sparse_mode,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    workspace=workspace,
                    out=[attn_output, softmax_lse],
                )
                torch.npu.graph_task_update_end(stream)
                event.record(stream)

    @staticmethod
    def update_graph_params(
        update_stream: Any,
        forward_context: Any,
        num_tokens: int,
        vllm_config: Any,
        speculative_config: Any = None,
        num_dcp_pcp_tokens: Any = None,
        draft_attn_metadatas: Any = None,
    ) -> None:
        if _EXTRA_CTX.is_draft_model or _EXTRA_CTX.sinks:
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )

        graph_params = acl_graph_module.get_graph_params()
        captured = graph_params.attn_params.get(num_tokens, ())
        handles = graph_params.handles.get(num_tokens, ())
        events = graph_params.events.get(num_tokens, ())
        attn_metadata = forward_context.attn_metadata
        if (
            len(captured) < workers
            or len(captured) != len(handles)
            or len(captured) != len(events)
            or not attn_metadata
            or hasattr(vllm_config.model_config.hf_text_config, "sliding_window")
        ):
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )

        metadata_keys = sorted(attn_metadata, key=_layer_index)
        if len(metadata_keys) < len(captured):
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )

        items: list[tuple[Any, ...]] = []
        for index, (param, handle, event) in enumerate(zip(captured, handles, events, strict=True)):
            if len(param) not in (21, 22):
                break
            normalized = (*param, None) if len(param) == 21 else param
            c8_k_scale = normalized[16]
            c8_v_scale = normalized[18]
            layer_name = normalized[20]
            adaptive_decision = normalized[21]
            metadata_key = (
                layer_name
                if layer_name is not None and layer_name in attn_metadata
                else metadata_keys[index]
            )
            metadata = attn_metadata[metadata_key]
            if (
                c8_k_scale is not None
                or c8_v_scale is not None
                or adaptive_decision is not None
                or metadata.tree_attn
                or not metadata.causal
            ):
                break
            items.append(
                (
                    param,
                    handle,
                    event,
                    metadata.seq_lens_list,
                    metadata.actual_seq_lengths_q,
                    metadata.block_tables,
                )
            )
        if len(items) != len(captured):
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )

        device, streams = state.ensure_streams(torch)
        gate = torch.npu.Event()
        gate.record(update_stream)
        chunks = [items[offset::workers] for offset in range(workers)]
        futures = [
            state.executor.submit(
                update_one_stream,
                device,
                stream,
                gate,
                chunk,
                graph_params.workspaces.get(num_tokens),
            )
            for stream, chunk in zip(streams, chunks, strict=True)
            if chunk
        ]
        for future in futures:
            future.result()

    impl.update_graph_params = update_graph_params
    setattr(impl, PATCH_MARKER, True)
    return True
