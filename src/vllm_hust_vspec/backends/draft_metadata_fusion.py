"""Opt-in Qwen3.5 MTP continuation metadata fusion.

The host proposer builds this metadata *before* draft graph replay.  Graph
capture takes a separate dummy_run path; the persistent slot/seq/query group
buffers written here retain the addresses registered by that path.  Positions
receive a fresh allocation on each call, matching the host clone lifetime for
async scheduling.  The kernel deliberately does not touch the CPU mirrors.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=(
        "batch_size",
        "seq_source_len",
        "seq_result_len",
        "query_result_len",
        "slot_capacity",
        "seq_capacity",
        "query_capacity",
        "position_capacity",
        "block_row_stride",
        "block_col_stride",
        "block_columns",
        "padding_slot_id",
    )
)
def _q35_mtp_continuation_metadata_kernel(
    used_positions_ptr,
    seq_source_ptr,
    position_source_ptr,
    block_table_ptr,
    slot_result_ptr,
    seq_result_ptr,
    query_result_ptr,
    position_result_ptr,
    batch_size,
    seq_source_len,
    seq_result_len,
    query_result_len,
    slot_capacity,
    seq_capacity,
    query_capacity,
    position_capacity,
    block_row_stride,
    block_col_stride,
    block_columns,
    block_size: tl.constexpr,
    max_model_len: tl.constexpr,
    padding_slot_id,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # used_update_positions is a contiguous [3, batch_size] M-RoPE copy.
    # All three axes advance; the first axis alone determines the KV slot.
    new_position = tl.load(
        used_positions_ptr + row,
        mask=row < 3 * batch_size,
        other=0,
    ) + 1
    tl.store(
        used_positions_ptr + row,
        new_position,
        mask=row < 3 * batch_size,
    )
    exceeds_limit = new_position >= max_model_len
    clamped_position = tl.where(exceeds_limit, 0, new_position)

    source_seq = tl.load(
        seq_source_ptr + row,
        mask=row < seq_source_len,
        other=0,
    )
    updated_seq = tl.where(row < batch_size, source_seq + 1, source_seq)
    updated_seq = tl.where(
        (row < batch_size) & (updated_seq > max_model_len),
        1,
        updated_seq,
    )
    tl.store(
        seq_result_ptr + row,
        tl.where(row < seq_result_len, updated_seq, 0),
        mask=row < seq_capacity,
    )

    # Both FULL and PIECEWISE set the first continuation's query starts to
    # arange; later continuations inherit exactly that same vector.
    tl.store(
        query_result_ptr + row,
        tl.where(row < query_result_len, row, 0),
        mask=row < query_capacity,
    )

    source_position = tl.load(
        position_source_ptr + row,
        mask=row < position_capacity,
        other=0,
    )
    tl.store(
        position_result_ptr + row,
        tl.where(row < batch_size, clamped_position, source_position),
        mask=row < position_capacity,
    )

    block_number = clamped_position // block_size
    block_id = tl.load(
        block_table_ptr + row * block_row_stride + block_number * block_col_stride,
        mask=(row < batch_size)
        & (~exceeds_limit)
        & (clamped_position >= 0)
        & (block_number >= 0)
        & (block_number < block_columns),
        other=0,
    )
    slot = block_id * block_size + clamped_position % block_size
    valid_block = (
        (clamped_position >= 0)
        & (block_number >= 0)
        & (block_number < block_columns)
    )
    slot = tl.where((~exceeds_limit) & valid_block, slot, padding_slot_id)
    tl.store(
        slot_result_ptr + row,
        tl.where(row < batch_size, slot, padding_slot_id),
        mask=row < slot_capacity,
    )


def can_fuse_q35_mtp_continuation(
    proposer: Any,
    arguments: dict[str, Any],
    *,
    full_mode: bool,
    piecewise_mode: bool,
    capturing: bool,
    spec_decode_state: Any,
) -> bool:
    """Cheap, value-free guard; never synchronizes a device tensor to host."""
    if not (
        getattr(proposer, "_vspec_q35_metadata_runtime", False)
        and (full_mode or piecewise_mode)
        and not capturing
        and proposer.method == "mtp"
        and proposer.uses_mrope
        and proposer.num_speculative_tokens in (4, 6, 8, 10, 12)
        and not proposer.parallel_drafting
        and not proposer.use_compress
        and proposer.sliding_window is None
        and getattr(proposer, "dcp_size", 1) == 1
        and getattr(proposer.runner, "dcp_manager", None) is None
        and not getattr(proposer.runner, "sparse_kv_offload_enabled", False)
        and len(proposer.draft_attn_groups) == 1
        and arguments["attn_group"] is proposer.draft_attn_groups[0]
        and not getattr(proposer, "_compact_merged_continuations", False)
        and not getattr(proposer, "_compact_piecewise_force", False)
    ):
        return False

    draft_index = arguments["draft_index"]
    batch_size = arguments["batch_size"]
    input_batch_size = arguments["input_batch_size"]
    if not (
        isinstance(draft_index, int)
        and isinstance(batch_size, int)
        and isinstance(input_batch_size, int)
        and 1 <= draft_index < len(proposer.seq_lens_group)
        and 0 < batch_size <= min(input_batch_size, 16)
    ):
        return False
    old = arguments["old_common_metadata"]
    if getattr(old, "attn_state", None) != spec_decode_state:
        return False
    # The non-DCP model runner reports zero prefill/decode counts even for a
    # mixed batch.  A uniform graph descriptor only means every request has
    # the same scheduled token count; a chunked prefill can also satisfy it.
    # The runner builds this bool CPU tensor from
    # num_computed_tokens_cpu < num_prompt_tokens_cpu before draft proposal.
    # Never inspect a device tensor here: that would introduce a D2H sync.
    is_prefilling = getattr(old, "is_prefilling", None)
    if not (
        isinstance(is_prefilling, torch.Tensor)
        and is_prefilling.device.type == "cpu"
        and is_prefilling.dtype == torch.bool
        and is_prefilling.ndim == 1
        and is_prefilling.numel() >= batch_size
    ):
        return False
    if bool(is_prefilling[:batch_size].any().item()):
        return False
    used = arguments["used_update_positions"]
    group = proposer.seq_lens_group[draft_index]
    query_group = proposer.query_start_loc_group[draft_index]
    slot_group = proposer.slot_mapping_group[draft_index]
    seq_result_len = input_batch_size if full_mode and draft_index == 1 else old.seq_lens.numel()
    query_result_len = (
        input_batch_size + 1 if full_mode else batch_size + 1
    )
    if draft_index > 1:
        query_result_len = old.query_start_loc.numel()

    block_size = proposer.kernel_block_size if proposer.has_gdn else proposer.block_size
    block_table = old.block_table_tensor
    expected_device = group.device
    return bool(
        0 < block_size
        and old.seq_lens.numel() >= batch_size
        and old.positions.numel() >= batch_size
        and seq_result_len >= batch_size
        and seq_result_len <= group.numel()
        and query_result_len <= query_group.numel()
        and batch_size <= slot_group.numel()
        and old.seq_lens.ndim == 1
        and old.seq_lens.stride(0) == 1
        and old.positions.ndim == 1
        and old.positions.stride(0) == 1
        and used.shape == (3, batch_size)
        and used.is_contiguous()
        and block_table.ndim == 2
        and block_table.shape[0] >= batch_size
        and block_table.shape[1] > (proposer.max_model_len - 1) // block_size
        and all(
            tensor.dtype == torch.int32 and tensor.device == expected_device
            for tensor in (
                used,
                old.seq_lens,
                block_table,
                group,
                query_group,
                slot_group,
            )
        )
        and old.positions.dtype in (torch.int32, torch.int64)
        and old.positions.device == expected_device
        # The host uses an advanced-indexed M-RoPE copy.  Reject unusual
        # callers that alias either source with an in-place kernel destination.
        and used.untyped_storage().data_ptr()
        != old.positions.untyped_storage().data_ptr()
        and old.seq_lens.untyped_storage().data_ptr()
        != group.untyped_storage().data_ptr()
    )


def fuse_q35_mtp_continuation(
    proposer: Any,
    arguments: dict[str, Any],
    *,
    full_mode: bool,
    attention_state: Any,
    padding_slot_id: int,
    precomputed_position: torch.Tensor | None = None,
    skip_kernel: bool = False,
) -> tuple[Any, Any]:
    """Mirror the host function's scalar/CPU work; fuse its device writes."""
    old = arguments["old_common_metadata"]
    draft_index = arguments["draft_index"]
    batch_size = arguments["batch_size"]
    input_batch_size = arguments["input_batch_size"]
    used = arguments["used_update_positions"]
    attn_group = arguments["attn_group"]
    common = proposer.shallow_copy_metadata(old)

    if draft_index == 1:
        if full_mode:
            common.num_reqs = input_batch_size
            common.block_table_tensor = proposer._adjust_tensor(
                old.block_table_tensor, input_batch_size
            )
        query_result_len = input_batch_size + 1 if full_mode else batch_size + 1
        common.query_start_loc_cpu = torch.from_numpy(
            proposer.token_arange_np[:query_result_len]
        ).clone()
        common.num_actual_tokens = batch_size
        common.max_query_len = 1
        common.decode_token_per_req = 1
        common.attn_state = attention_state
        common.graph_pad_size = -1
        common.num_input_tokens = input_batch_size
    else:
        query_result_len = old.query_start_loc.numel()

    seq_result_len = input_batch_size if full_mode and draft_index == 1 else old.seq_lens.numel()

    def clone_cpu(source: torch.Tensor | None, *, clamp: bool) -> torch.Tensor | None:
        if source is None:
            return None
        if full_mode and draft_index == 1:
            source = proposer._adjust_tensor(source, input_batch_size)
        result = source.clone()
        result[:batch_size].add_(1)
        if clamp:
            result[:batch_size].masked_fill_(
                result[:batch_size] > proposer.max_model_len, 1
            )
        return result

    common.seq_lens_cpu = clone_cpu(old.seq_lens_cpu, clamp=True)
    common._seq_lens_cpu = clone_cpu(old._seq_lens_cpu, clamp=True)
    common.num_computed_tokens_cpu = clone_cpu(
        old.num_computed_tokens_cpu, clamp=False
    )

    slot_result = proposer.slot_mapping_group[draft_index]
    seq_result = proposer.seq_lens_group[draft_index]
    query_result = proposer.query_start_loc_group[draft_index]
    # Do not recycle this buffer across async scheduling steps.  The host
    # implementation clones positions for exactly this lifetime guarantee.
    if skip_kernel:
        if precomputed_position is None or precomputed_position.shape != old.positions.shape:
            raise RuntimeError("batched Q35 MTP position buffer missing or shape changed")
        position_result = precomputed_position
    else:
        position_result = torch.empty_like(old.positions)
    block_size = proposer.kernel_block_size if proposer.has_gdn else proposer.block_size
    max_rows = max(
        3 * batch_size,
        slot_result.numel(),
        seq_result.numel(),
        query_result.numel(),
        position_result.numel(),
    )
    if not skip_kernel:
        _q35_mtp_continuation_metadata_kernel[(triton.cdiv(max_rows, 256),)](
            used,
            old.seq_lens,
            old.positions,
            old.block_table_tensor,
            slot_result,
            seq_result,
            query_result,
            position_result,
            batch_size,
            old.seq_lens.numel(),
            seq_result_len,
            query_result_len,
            slot_result.numel(),
            seq_result.numel(),
            query_result.numel(),
            position_result.numel(),
            old.block_table_tensor.stride(0),
            old.block_table_tensor.stride(1),
            old.block_table_tensor.shape[1],
            block_size,
            proposer.max_model_len,
            padding_slot_id,
            BLOCK=256,
            num_warps=4,
        )
    common.slot_mapping = slot_result
    common.seq_lens = seq_result[:seq_result_len]
    common.query_start_loc = query_result[:query_result_len]
    common.positions = position_result
    attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
        common, draft_index
    )
    return common, attn_metadata
