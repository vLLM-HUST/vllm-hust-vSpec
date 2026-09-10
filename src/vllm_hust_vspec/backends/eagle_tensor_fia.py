"""Tensor-backed sequence lengths for stable EAGLE FULL graph replay."""

from __future__ import annotations

from typing import Any

PATCH_MARKER = "_vllm_hust_vspec_tensor_seq_lens_patched"


def apply_tensor_seq_lens_patch() -> bool:
    """Avoid FIA task re-tiling when all dynamic inputs are device tensors."""
    import torch
    import torch_npu
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.compilation import acl_graph as acl_graph_module
    from vllm_ascend.compilation.acl_graph import _EXTRA_CTX

    impl = AscendAttentionBackendImpl
    if getattr(impl, PATCH_MARKER, False):
        return False

    original_forward = impl.forward_fused_infer_attention
    original_update = impl.update_graph_params

    def forward_fused_infer_attention(
        self: Any,
        query: Any,
        key: Any,
        value: Any,
        attn_metadata: Any,
        output: Any,
        kv_cache: Any = None,
    ) -> Any:
        if _EXTRA_CTX.capturing and not _EXTRA_CTX.is_draft_model and self.sinks is None:
            attn_output, num_tokens = self.full_graph_fia_v2(
                query,
                key,
                value,
                attn_metadata,
                output,
            )
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        return original_forward(
            self,
            query,
            key,
            value,
            attn_metadata,
            output,
            kv_cache,
        )

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
        graph_params = (
            acl_graph_module.get_draft_graph_params()
            if _EXTRA_CTX.is_draft_model
            else acl_graph_module.get_graph_params()
        )
        captured = graph_params.attn_params.get(num_tokens, ())
        if not captured or len(captured[0]) != 15:
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )

        if _EXTRA_CTX.is_draft_model:
            # Serial EAGLE has one metadata set per Draft step. The initial
            # version only accelerates Target replay, where the benefit and
            # pointer-stability conditions are both clear.
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )

        attn_metadata = forward_context.attn_metadata
        attn_keys = list(attn_metadata)
        captured_count = len(captured)
        if not attn_keys or captured_count == 0:
            return
        if captured_count > len(attn_keys):
            repeats = (captured_count + len(attn_keys) - 1) // len(attn_keys)
            attn_keys = (attn_keys * repeats)[:captured_count]
        else:
            attn_keys = attn_keys[:captured_count]

        # FIA v2 captures KV lengths as a tensor. Stable speculative decode
        # also has a fixed cumulative query layout, so replay only needs the
        # ExternalEvents that release the already captured operators. If a
        # pointer or query layout differs, rebind that layer through the native
        # graph-task update API.
        with torch.npu.stream(update_stream):
            for key, param, handle, event in zip(
                attn_keys,
                captured,
                graph_params.handles[num_tokens],
                graph_params.events[num_tokens],
                strict=True,
            ):
                (
                    query,
                    key_cache,
                    value_cache,
                    captured_block_table,
                    attn_mask,
                    block_size,
                    captured_seq_lens,
                    num_kv_heads,
                    num_heads,
                    scale,
                    sliding_window,
                    sinks,
                    attn_output,
                    softmax_lse,
                    layer_name,
                ) = param
                metadata_key = (
                    layer_name if layer_name is not None and layer_name in attn_metadata else key
                )
                metadata = attn_metadata[metadata_key]
                query_ends = metadata.actual_seq_lengths_q
                query_starts = [0, *query_ends[:-1]]
                query_widths = [
                    end - start
                    for start, end in zip(
                        query_starts,
                        query_ends,
                        strict=True,
                    )
                ]
                uniform_query = bool(query_widths) and len(set(query_widths)) == 1
                pointers_stable = (
                    captured_seq_lens.data_ptr() == metadata.seq_lens.data_ptr()
                    and captured_block_table.data_ptr() == metadata.block_tables.data_ptr()
                )
                if not (uniform_query and pointers_stable):
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query,
                        key=key_cache,
                        value=value_cache,
                        atten_mask=attn_mask,
                        block_table=metadata.block_tables,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_qlen=query_ends,
                        actual_seq_kvlen=metadata.seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        sparse_mode=(4 if sliding_window is not None else 3),
                        pre_tokens=(
                            sliding_window if sliding_window is not None else 2_147_483_647
                        ),
                        next_tokens=0,
                        softmax_scale=scale,
                        learnable_sink=sinks,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)

    impl.forward_fused_infer_attention = forward_fused_infer_attention
    impl.update_graph_params = update_graph_params
    setattr(impl, PATCH_MARKER, True)
    return True
