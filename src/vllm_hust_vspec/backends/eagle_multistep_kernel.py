"""HOLD: one-launch Q35 Eagle-MTP continuation metadata kernel.

The module is private and opt-in.  It is syntax/CPU-mock tested only; neither
Triton NPU lowering nor graph replay has been validated.  Never enable it in
the running benchmark without a separate parity gate.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _write_step(
    row,
    used_initial,
    seq_initial,
    position_initial,
    block_table_ptr,
    slot_ptr,
    seq_ptr,
    query_ptr,
    position_ptr,
    batch_size,
    seq_result_len,
    query_result_len,
    slot_capacity,
    seq_capacity,
    query_capacity,
    position_capacity,
    block_row_stride,
    block_col_stride,
    block_columns,
    padding_slot_id,
    STEP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_MODEL_LEN: tl.constexpr,
):
    new_position = used_initial + STEP
    exceeds = new_position >= MAX_MODEL_LEN
    clamped = tl.where(exceeds, 0, new_position)

    seq = seq_initial
    for _ in tl.static_range(STEP):
        next_seq = seq + 1
        seq = tl.where(row < batch_size, tl.where(next_seq > MAX_MODEL_LEN, 1, next_seq), seq)
    tl.store(
        seq_ptr + row,
        tl.where(row < seq_result_len, seq, 0),
        mask=row < seq_capacity,
    )
    tl.store(
        query_ptr + row,
        tl.where(row < query_result_len, row, 0),
        mask=row < query_capacity,
    )
    tl.store(
        position_ptr + row,
        tl.where(row < batch_size, clamped, position_initial),
        mask=row < position_capacity,
    )

    block_number = clamped // BLOCK_SIZE
    valid = (clamped >= 0) & (block_number >= 0) & (block_number < block_columns)
    block_id = tl.load(
        block_table_ptr + row * block_row_stride + block_number * block_col_stride,
        mask=(row < batch_size) & (~exceeds) & valid,
        other=0,
    )
    slot = block_id * BLOCK_SIZE + clamped % BLOCK_SIZE
    slot = tl.where((~exceeds) & valid, slot, padding_slot_id)
    tl.store(
        slot_ptr + row,
        tl.where(row < batch_size, slot, padding_slot_id),
        mask=row < slot_capacity,
    )


@triton.jit
def _q35_eagle_multistep_metadata_kernel(
    used_ptr,
    seq_source_ptr,
    position_source_ptr,
    block_table_ptr,
    slot_1, seq_1, query_1, position_1,
    slot_2, seq_2, query_2, position_2,
    slot_3, seq_3, query_3, position_3,
    slot_4, seq_4, query_4, position_4,
    slot_5, seq_5, query_5, position_5,
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
    padding_slot_id,
    STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_MODEL_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    used_initial = tl.load(used_ptr + row, mask=row < 3 * batch_size, other=0)
    seq_initial = tl.load(seq_source_ptr + row, mask=row < seq_source_len, other=0)
    position_initial = tl.load(position_source_ptr + row, mask=row < position_capacity, other=0)

    if STEPS >= 1:
        _write_step(row, used_initial, seq_initial, position_initial, block_table_ptr,
                    slot_1, seq_1, query_1, position_1, batch_size,
                    seq_result_len, query_result_len, slot_capacity, seq_capacity,
                    query_capacity, position_capacity, block_row_stride,
                    block_col_stride, block_columns, padding_slot_id,
                    1, BLOCK_SIZE, MAX_MODEL_LEN)
    if STEPS >= 2:
        _write_step(row, used_initial, seq_initial, position_initial, block_table_ptr,
                    slot_2, seq_2, query_2, position_2, batch_size,
                    seq_result_len, query_result_len, slot_capacity, seq_capacity,
                    query_capacity, position_capacity, block_row_stride,
                    block_col_stride, block_columns, padding_slot_id,
                    2, BLOCK_SIZE, MAX_MODEL_LEN)
    if STEPS >= 3:
        _write_step(row, used_initial, seq_initial, position_initial, block_table_ptr,
                    slot_3, seq_3, query_3, position_3, batch_size,
                    seq_result_len, query_result_len, slot_capacity, seq_capacity,
                    query_capacity, position_capacity, block_row_stride,
                    block_col_stride, block_columns, padding_slot_id,
                    3, BLOCK_SIZE, MAX_MODEL_LEN)
    if STEPS >= 4:
        _write_step(row, used_initial, seq_initial, position_initial, block_table_ptr,
                    slot_4, seq_4, query_4, position_4, batch_size,
                    seq_result_len, query_result_len, slot_capacity, seq_capacity,
                    query_capacity, position_capacity, block_row_stride,
                    block_col_stride, block_columns, padding_slot_id,
                    4, BLOCK_SIZE, MAX_MODEL_LEN)
    if STEPS >= 5:
        _write_step(row, used_initial, seq_initial, position_initial, block_table_ptr,
                    slot_5, seq_5, query_5, position_5, batch_size,
                    seq_result_len, query_result_len, slot_capacity, seq_capacity,
                    query_capacity, position_capacity, block_row_stride,
                    block_col_stride, block_columns, padding_slot_id,
                    5, BLOCK_SIZE, MAX_MODEL_LEN)
    tl.store(used_ptr + row, used_initial + STEPS, mask=row < 3 * batch_size)


def launch_multistep(proposer: Any, old: Any, used: torch.Tensor, *, batch_size: int,
                     input_batch_size: int, full_mode: bool,
                     padding_slot_id: int, positions: list[torch.Tensor]) -> None:
    """Host launcher; caller owns preflight and the proposal-level state."""
    gamma = proposer.num_speculative_tokens
    assert gamma in (4, 6) and len(positions) == gamma - 1
    slot_groups = proposer.slot_mapping_group
    seq_groups = proposer.seq_lens_group
    query_groups = proposer.query_start_loc_group
    active = [
        (slot_groups[i], seq_groups[i], query_groups[i], positions[i - 1])
        for i in range(1, gamma)
    ]
    # The JIT signature has five statically named destinations.  Compile-time
    # `STEPS=3` eliminates slots 4/5 for gamma 4, so dummy pointers are unused.
    while len(active) < 5:
        active.append(active[-1])
    flat = [tensor for group in active for tensor in group]
    seq_result_len = input_batch_size if full_mode else old.seq_lens.numel()
    query_result_len = input_batch_size + 1 if full_mode else batch_size + 1
    max_rows = max(
        3 * batch_size,
        active[0][0].numel(),
        active[0][1].numel(),
        active[0][2].numel(),
        old.positions.numel(),
    )
    block_size = proposer.kernel_block_size if proposer.has_gdn else proposer.block_size
    _q35_eagle_multistep_metadata_kernel[(triton.cdiv(max_rows, 256),)](
        used, old.seq_lens, old.positions, old.block_table_tensor,
        *flat,
        batch_size, old.seq_lens.numel(), seq_result_len, query_result_len,
        active[0][0].numel(), active[0][1].numel(), active[0][2].numel(),
        old.positions.numel(), old.block_table_tensor.stride(0),
        old.block_table_tensor.stride(1), old.block_table_tensor.shape[1],
        padding_slot_id,
        STEPS=gamma - 1, BLOCK_SIZE=block_size,
        MAX_MODEL_LEN=proposer.max_model_len, BLOCK=256, num_warps=4,
    )
