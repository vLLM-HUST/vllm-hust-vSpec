"""Low-overhead repetition-penalty helpers for serial Draft on Ascend."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available


@triton.jit
def _apply_unique_history_repetition_kernel(
    logits_ptr,
    history_ptr,
    token_to_logit_ptr,
    repetition_penalties_ptr,
    logits_row_stride,
    history_row_stride,
    num_rows,
    history_width,
    token_id_map_size,
    logits_vocab_size,
    HISTORY_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * HISTORY_BLOCK + tl.arange(0, HISTORY_BLOCK)
    valid_position = (row < num_rows) & (offsets < history_width)
    token_ids = tl.load(
        history_ptr + row * history_row_stride + offsets,
        mask=valid_position,
        other=-1,
    )
    valid_token = valid_position & (token_ids >= 0) & (token_ids < token_id_map_size)
    safe_token_ids = tl.where(valid_token, token_ids, 0)
    logit_ids = tl.load(
        token_to_logit_ptr + safe_token_ids,
        mask=valid_token,
        other=-1,
    )
    valid_token &= (logit_ids >= 0) & (logit_ids < logits_vocab_size)
    safe_logit_ids = tl.where(valid_token, logit_ids, 0)
    values = tl.load(
        logits_ptr + row * logits_row_stride + safe_logit_ids,
        mask=valid_token,
        other=0.0,
    ).to(tl.float32)
    penalty = tl.load(repetition_penalties_ptr + row, mask=row < num_rows, other=1.0)
    values *= tl.where(values > 0, 1.0 / penalty, penalty)
    tl.store(
        logits_ptr + row * logits_row_stride + safe_logit_ids,
        values,
        mask=valid_token,
    )


@triton.jit
def _apply_prefix_repetition_kernel(
    logits_ptr,
    history_ptr,
    prefix_ptr,
    token_to_logit_ptr,
    repetition_penalties_ptr,
    logits_row_stride,
    history_row_stride,
    prefix_row_stride,
    num_rows,
    history_width,
    prefix_position: tl.constexpr,
    token_id_map_size,
    logits_vocab_size,
    HISTORY_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token_id = tl.load(prefix_ptr + row * prefix_row_stride + prefix_position)
    offsets = tl.arange(0, HISTORY_BLOCK)
    history_ids = tl.load(
        history_ptr + row * history_row_stride + offsets,
        mask=(row < num_rows) & (offsets < history_width),
        other=-1,
    )
    seen = tl.sum((history_ids == token_id).to(tl.int32), axis=0) > 0
    for earlier_position in tl.static_range(0, prefix_position):
        seen |= tl.load(prefix_ptr + row * prefix_row_stride + earlier_position) == token_id
    valid_token_id = (row < num_rows) & (token_id >= 0) & (token_id < token_id_map_size) & ~seen
    safe_token_id = tl.where(valid_token_id, token_id, 0)
    logit_id = tl.load(
        token_to_logit_ptr + safe_token_id,
        mask=valid_token_id,
        other=-1,
    )
    valid_token = valid_token_id & (logit_id >= 0) & (logit_id < logits_vocab_size)
    safe_logit_id = tl.where(valid_token, logit_id, 0)
    value = tl.load(
        logits_ptr + row * logits_row_stride + safe_logit_id,
        mask=valid_token,
        other=0.0,
    ).to(tl.float32)
    penalty = tl.load(repetition_penalties_ptr + row, mask=row < num_rows, other=1.0)
    value *= tl.where(value > 0, 1.0 / penalty, penalty)
    tl.store(
        logits_ptr + row * logits_row_stride + safe_logit_id,
        value,
        mask=valid_token,
    )


class DraftRepetitionState:
    """Persistent graph inputs for Draft-side repetition alignment."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        history_width: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.history_width = history_width
        self.vocab_size = vocab_size
        self.history = torch.full(
            (max_batch_size, history_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.repetition_penalties = torch.ones(
            max_batch_size,
            dtype=torch.float32,
            device=device,
        )
        self.token_to_logit = torch.arange(
            vocab_size,
            dtype=torch.int32,
            device=device,
        )
        self._active_ids_address: int | None = None
        self.history_cpu = torch.full(
            (max_batch_size, history_width),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self.penalties_cpu = torch.ones(
            max_batch_size,
            dtype=torch.float32,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )

    def configure_active_ids(self, active_ids: torch.Tensor | None) -> None:
        if active_ids is None:
            return
        active_ids_address = active_ids.data_ptr()
        if self._active_ids_address == active_ids_address:
            return
        self.token_to_logit.fill_(-1)
        compact_ids = torch.arange(
            active_ids.numel(),
            dtype=torch.int32,
            device=active_ids.device,
        )
        valid = active_ids < self.vocab_size
        self.token_to_logit[active_ids[valid].long()] = compact_ids[valid]
        self._active_ids_address = active_ids_address

    def update(
        self,
        *,
        prompt_token_ids: Any,
        prompt_lengths: Any,
        output_token_ids: list[list[int]],
        repetition_penalties: Any,
        num_rows: int,
        vocab_size: int,
    ) -> bool:
        if num_rows > self.max_batch_size:
            raise RuntimeError(
                "Draft repetition batch exceeds its persistent buffer: "
                f"{num_rows} > {self.max_batch_size}"
            )

        history_rows, truncated = _collect_unique_history(
            prompt_token_ids=prompt_token_ids,
            prompt_lengths=prompt_lengths,
            output_token_ids=output_token_ids,
            num_rows=num_rows,
            vocab_size=vocab_size,
            history_width=self.history_width,
        )
        self.history_cpu.fill_(-1)
        for row, token_ids in enumerate(history_rows):
            if token_ids:
                self.history_cpu[row, : len(token_ids)] = torch.tensor(
                    token_ids,
                    dtype=torch.int32,
                )

        self.penalties_cpu.fill_(1.0)
        if isinstance(repetition_penalties, torch.Tensor):
            penalties = repetition_penalties.detach().to(
                device="cpu",
                dtype=torch.float32,
            )
            self.penalties_cpu[:num_rows].copy_(penalties[:num_rows])
        else:
            self.penalties_cpu[:num_rows] = torch.as_tensor(
                repetition_penalties[:num_rows],
                dtype=torch.float32,
            )

        self.history.copy_(self.history_cpu, non_blocking=True)
        self.repetition_penalties.copy_(self.penalties_cpu, non_blocking=True)
        return truncated


def _collect_unique_history(
    *,
    prompt_token_ids: Any,
    prompt_lengths: Any,
    output_token_ids: list[list[int]],
    num_rows: int,
    vocab_size: int,
    history_width: int,
) -> tuple[list[list[int]], bool]:
    rows: list[list[int]] = []
    truncated = False
    for row in range(num_rows):
        prompt_length = int(prompt_lengths[row])
        prompt = prompt_token_ids[row][:prompt_length]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        output = output_token_ids[row] if row < len(output_token_ids) else []
        unique_ids = dict.fromkeys(
            int(token_id) for token_id in (*prompt, *output) if 0 <= int(token_id) < vocab_size
        )
        token_ids = list(unique_ids)
        if len(token_ids) > history_width:
            truncated = True
            token_ids = token_ids[-history_width:]
        rows.append(token_ids)
    return rows, truncated


def apply_draft_repetition_penalty(
    logits: Any,
    state: DraftRepetitionState,
    prefix_token_ids: list[torch.Tensor],
) -> Any:
    """Apply Target-compatible repetition penalty before Draft greedy sampling."""
    logits_tensor = getattr(logits, "logits", logits)
    if not isinstance(logits_tensor, torch.Tensor):
        raise TypeError("Draft repetition alignment requires tensor-backed logits")
    num_rows = logits_tensor.shape[0]
    history_block = 256
    num_history_blocks = (state.history_width + history_block - 1) // history_block
    _apply_unique_history_repetition_kernel[(num_rows, num_history_blocks)](
        logits_tensor,
        state.history,
        state.token_to_logit,
        state.repetition_penalties,
        logits_tensor.stride(0),
        state.history.stride(0),
        num_rows,
        state.history_width,
        state.vocab_size,
        logits_tensor.shape[-1],
        HISTORY_BLOCK=history_block,
    )

    if prefix_token_ids:
        prefix = torch.stack(prefix_token_ids, dim=1)
        prefix_history_block = triton.next_power_of_2(state.history_width)
        for prefix_position in range(len(prefix_token_ids)):
            _apply_prefix_repetition_kernel[(num_rows,)](
                logits_tensor,
                state.history,
                prefix,
                state.token_to_logit,
                state.repetition_penalties,
                logits_tensor.stride(0),
                state.history.stride(0),
                prefix.stride(0),
                num_rows,
                state.history_width,
                prefix_position,
                state.vocab_size,
                logits_tensor.shape[-1],
                HISTORY_BLOCK=prefix_history_block,
            )
    return logits


@triton.jit
def _build_seen_mask_kernel(
    token_ids_ptr,
    seen_mask_ptr,
    token_ids_row_stride,
    seen_mask_row_stride,
    seq_len,
    vocab_size,
    SEQ_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    valid_position = offsets < seq_len
    token_ids = tl.load(
        token_ids_ptr + row * token_ids_row_stride + offsets,
        mask=valid_position,
        other=vocab_size,
    )
    valid_token = valid_position & (token_ids >= 0) & (token_ids < vocab_size)
    packed_indices = token_ids // 32
    bit_values = tl.full((SEQ_BLOCK,), 1, tl.int32) << (token_ids % 32)
    tl.atomic_or(
        seen_mask_ptr + row * seen_mask_row_stride + packed_indices,
        bit_values,
        mask=valid_token,
    )


@triton.jit
def _append_seen_tokens_kernel(
    seen_mask_ptr,
    slot_indices_ptr,
    token_ids_ptr,
    seen_mask_row_stride,
    num_updates,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_position = offsets < num_updates
    token_ids = tl.load(
        token_ids_ptr + offsets,
        mask=valid_position,
        other=vocab_size,
    )
    slot_indices = tl.load(
        slot_indices_ptr + offsets,
        mask=valid_position,
        other=0,
    )
    valid_token = valid_position & (token_ids >= 0) & (token_ids < vocab_size)
    tl.store(
        seen_mask_ptr + slot_indices * seen_mask_row_stride + token_ids,
        1,
        mask=valid_token,
    )


@triton.jit
def _append_seen_tokens_and_history_kernel(
    seen_mask_ptr,
    history_ptr,
    slot_indices_ptr,
    token_ids_ptr,
    history_positions_ptr,
    seen_mask_row_stride,
    history_row_stride,
    num_updates,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_position = offsets < num_updates
    token_ids = tl.load(
        token_ids_ptr + offsets,
        mask=valid_position,
        other=vocab_size,
    )
    slot_indices = tl.load(
        slot_indices_ptr + offsets,
        mask=valid_position,
        other=0,
    )
    history_positions = tl.load(
        history_positions_ptr + offsets,
        mask=valid_position,
        other=0,
    )
    valid_token = valid_position & (token_ids >= 0) & (token_ids < vocab_size)
    tl.store(
        seen_mask_ptr + slot_indices * seen_mask_row_stride + token_ids,
        1,
        mask=valid_token,
    )
    tl.store(
        history_ptr + slot_indices * history_row_stride + history_positions,
        token_ids,
        mask=valid_token,
    )


@triton.jit
def _apply_sparse_history_repetition_kernel(
    logits_ptr,
    history_ptr,
    history_counts_ptr,
    request_slots_ptr,
    repetition_penalties_ptr,
    logits_row_stride,
    history_row_stride,
    num_rows,
    vocab_size,
    HISTORY_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    request_slot = tl.load(request_slots_ptr + row, mask=row < num_rows, other=0)
    history_count = tl.load(
        history_counts_ptr + request_slot,
        mask=row < num_rows,
        other=0,
    )
    offsets = block * HISTORY_BLOCK + tl.arange(0, HISTORY_BLOCK)
    valid_position = (row < num_rows) & (offsets < history_count)
    token_ids = tl.load(
        history_ptr + request_slot * history_row_stride + offsets,
        mask=valid_position,
        other=0,
    )
    valid_token = valid_position & (token_ids >= 0) & (token_ids < vocab_size)
    safe_token_ids = tl.where(valid_token, token_ids, 0)
    values = tl.load(
        logits_ptr + row * logits_row_stride + safe_token_ids,
        mask=valid_token,
        other=0.0,
    ).to(tl.float32)
    penalty = tl.load(
        repetition_penalties_ptr + row,
        mask=row < num_rows,
        other=1.0,
    )
    values *= tl.where(values > 0, 1.0 / penalty, penalty)
    tl.store(
        logits_ptr + row * logits_row_stride + safe_token_ids,
        values,
        mask=valid_token,
    )


@triton.jit
def _apply_sparse_prefix_repetition_kernel(
    logits_ptr,
    prefix_ptr,
    seen_mask_ptr,
    request_slots_ptr,
    repetition_penalties_ptr,
    logits_row_stride,
    seen_mask_row_stride,
    num_rows,
    vocab_size,
):
    row = tl.program_id(0)
    request_slot = tl.load(request_slots_ptr + row, mask=row < num_rows, other=0)
    token_id = tl.load(prefix_ptr + row, mask=row < num_rows, other=0)
    valid_token = (row < num_rows) & (token_id >= 0) & (token_id < vocab_size)
    safe_token_id = tl.where(valid_token, token_id, 0)
    already_seen = tl.load(
        seen_mask_ptr + request_slot * seen_mask_row_stride + safe_token_id,
        mask=valid_token,
        other=True,
    ).to(tl.int1)
    apply_penalty = valid_token & ~already_seen
    value = tl.load(
        logits_ptr + row * logits_row_stride + safe_token_id,
        mask=apply_penalty,
        other=0.0,
    ).to(tl.float32)
    penalty = tl.load(
        repetition_penalties_ptr + row,
        mask=row < num_rows,
        other=1.0,
    )
    value *= tl.where(value > 0, 1.0 / penalty, penalty)
    tl.store(
        logits_ptr + row * logits_row_stride + safe_token_id,
        value,
        mask=apply_penalty,
    )


@triton.jit
def _append_spec_prefix_kernel(
    row_seen_mask_ptr,
    repeat_indices_ptr,
    local_positions_ptr,
    spec_token_ids_ptr,
    row_seen_mask_stride,
    spec_row_stride,
    num_rows,
    vocab_size,
    MAX_SPEC_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    request_index = tl.load(repeat_indices_ptr + row)
    local_position = tl.load(local_positions_ptr + row)
    for position in tl.static_range(0, MAX_SPEC_LEN):
        spec_token = tl.load(
            spec_token_ids_ptr + request_index * spec_row_stride + position,
        )
        valid = (
            (row < num_rows)
            & (position < local_position)
            & (spec_token >= 0)
            & (spec_token < vocab_size)
        )
        tl.store(
            row_seen_mask_ptr + row * row_seen_mask_stride + spec_token,
            1,
            mask=valid,
        )


@triton.jit
def _apply_repetition_kernel(
    logits_ptr,
    row_seen_mask_ptr,
    row_penalties_ptr,
    logits_row_stride,
    row_seen_mask_stride,
    num_rows,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    program = tl.program_id(0)
    num_programs = tl.num_programs(0)
    rows_per_program = tl.cdiv(num_rows, num_programs)
    row_start = program * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, num_rows)

    for row in range(row_start, row_end):
        repetition_penalty = tl.load(row_penalties_ptr + row)
        for vocab_start in range(0, vocab_size, BLOCK_SIZE):
            offsets = vocab_start + tl.arange(0, BLOCK_SIZE)
            valid_vocab = offsets < vocab_size
            seen = tl.load(
                row_seen_mask_ptr + row * row_seen_mask_stride + offsets,
                mask=valid_vocab,
                other=False,
            ).to(tl.int1)

            logits = tl.load(
                logits_ptr + row * logits_row_stride + offsets,
                mask=valid_vocab,
                other=0.0,
            ).to(tl.float32)
            scale = tl.where(seen, repetition_penalty, 1.0)
            logits *= tl.where(logits > 0, 1.0 / scale, scale)
            tl.store(
                logits_ptr + row * logits_row_stride + offsets,
                logits,
                mask=valid_vocab,
            )


@triton.jit
def _repetition_argmax_partials_kernel(
    logits_ptr,
    vocab_ids_ptr,
    seen_mask_ptr,
    logits_indices_ptr,
    request_slots_ptr,
    repeat_indices_ptr,
    local_positions_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    repetition_penalties_ptr,
    partial_values_ptr,
    partial_ids_ptr,
    logits_row_stride,
    seen_mask_row_stride,
    partial_row_stride,
    vocab_size,
    token_id_limit,
    MAX_SPEC_LEN: tl.constexpr,
    USE_VOCAB_IDS: tl.constexpr,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    VOCAB_GRID_SIZE: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr,
):
    """Reduce one vocabulary tile after applying exact repetition penalty."""
    output_row = tl.program_id(0)
    vocab_program = tl.program_id(1)
    request_index = tl.load(repeat_indices_ptr + output_row)
    request_slot = tl.load(request_slots_ptr + request_index)
    local_position = tl.load(local_positions_ptr + output_row)
    request_start = tl.load(
        cu_num_draft_tokens_ptr + request_index - 1,
        mask=(request_index > 0) & (local_position > 0),
        other=0,
    )
    logits_row = tl.load(logits_indices_ptr + output_row)
    penalty = tl.load(repetition_penalties_ptr + request_index)

    best_value = float("-inf")
    best_token = token_id_limit
    for vocab_block in tl.range(
        vocab_program,
        NUM_VOCAB_BLOCKS,
        VOCAB_GRID_SIZE,
    ):
        offsets = vocab_block * VOCAB_BLOCK_SIZE + tl.arange(
            0,
            VOCAB_BLOCK_SIZE,
        )
        valid_vocab = offsets < vocab_size
        token_ids = (
            tl.load(
                vocab_ids_ptr + offsets,
                mask=valid_vocab,
                other=token_id_limit,
            ).to(tl.int32)
            if USE_VOCAB_IDS
            else offsets
        )
        valid_token = valid_vocab & (token_ids >= 0) & (token_ids < token_id_limit)
        seen = tl.load(
            seen_mask_ptr + request_slot * seen_mask_row_stride + token_ids,
            mask=valid_token,
            other=False,
        ).to(tl.int1)
        for position in tl.static_range(0, MAX_SPEC_LEN):
            spec_token = tl.load(
                draft_token_ids_ptr + request_start + position,
                mask=position < local_position,
                other=vocab_size,
            )
            seen |= (position < local_position) & (token_ids == spec_token)

        values = tl.load(
            logits_ptr + logits_row * logits_row_stride + offsets,
            mask=valid_token,
            other=float("-inf"),
        ).to(tl.float32)
        penalized = values * tl.where(values > 0, 1.0 / penalty, penalty)
        values = tl.where(seen, penalized, values)

        block_value = tl.max(values, axis=0)
        block_token = tl.min(
            tl.where(values == block_value, token_ids, token_id_limit),
            axis=0,
        )
        better = (block_value > best_value) | (
            (block_value == best_value) & (block_token < best_token)
        )
        best_value = tl.where(better, block_value, best_value)
        best_token = tl.where(better, block_token, best_token)

    partial_offset = output_row * partial_row_stride + vocab_program
    tl.store(partial_values_ptr + partial_offset, best_value)
    tl.store(
        partial_ids_ptr + partial_offset,
        best_token,
    )


@triton.jit
def _repetition_argmax_finalize_kernel(
    partial_values_ptr,
    partial_ids_ptr,
    output_ids_ptr,
    partial_row_stride,
    num_partials,
    PARTIAL_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, PARTIAL_BLOCK_SIZE)
    values = tl.load(
        partial_values_ptr + row * partial_row_stride + offsets,
        mask=offsets < num_partials,
        other=float("-inf"),
    )
    token_ids = tl.load(
        partial_ids_ptr + row * partial_row_stride + offsets,
        mask=offsets < num_partials,
        other=2147483647,
    )
    max_value = tl.max(values, axis=0)
    token_id = tl.min(
        tl.where(values == max_value, token_ids, 2147483647),
        axis=0,
    )
    tl.store(output_ids_ptr + row, token_id)


@triton.jit
def _repetition_argmax_margin_finalize_kernel(
    partial_values_ptr,
    partial_ids_ptr,
    output_ids_ptr,
    relaxed_mask_ptr,
    logits_ptr,
    seen_mask_ptr,
    logits_indices_ptr,
    request_slots_ptr,
    repeat_indices_ptr,
    local_positions_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    repetition_penalties_ptr,
    partial_row_stride,
    logits_row_stride,
    seen_mask_row_stride,
    num_partials,
    target_rows,
    vocab_size,
    max_margin,
    PARTIAL_BLOCK_SIZE: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, PARTIAL_BLOCK_SIZE)
    values = tl.load(
        partial_values_ptr + row * partial_row_stride + offsets,
        mask=offsets < num_partials,
        other=float("-inf"),
    )
    token_ids = tl.load(
        partial_ids_ptr + row * partial_row_stride + offsets,
        mask=offsets < num_partials,
        other=2147483647,
    )
    max_value = tl.max(values, axis=0)
    token_id = tl.min(
        tl.where(values == max_value, token_ids, 2147483647),
        axis=0,
    )
    tl.store(output_ids_ptr + row, token_id)
    is_target_row = row < target_rows
    request_index = tl.load(
        repeat_indices_ptr + row,
        mask=is_target_row,
        other=0,
    )
    request_slot = tl.load(
        request_slots_ptr + request_index,
        mask=is_target_row,
        other=0,
    )
    local_position = tl.load(
        local_positions_ptr + row,
        mask=is_target_row,
        other=0,
    )
    request_start = tl.load(
        cu_num_draft_tokens_ptr + request_index - 1,
        mask=is_target_row & (request_index > 0) & (local_position > 0),
        other=0,
    )
    draft_token = tl.load(
        draft_token_ids_ptr + row,
        mask=is_target_row,
        other=vocab_size,
    )
    valid_draft = is_target_row & (draft_token >= 0) & (draft_token < vocab_size)
    safe_draft = tl.where(valid_draft, draft_token, 0)
    seen = tl.load(
        seen_mask_ptr + request_slot * seen_mask_row_stride + safe_draft,
        mask=valid_draft,
        other=False,
    ).to(tl.int1)
    for position in tl.static_range(0, MAX_SPEC_LEN):
        prefix_token = tl.load(
            draft_token_ids_ptr + request_start + position,
            mask=is_target_row & (position < local_position),
            other=vocab_size,
        )
        seen |= (position < local_position) & (prefix_token == draft_token)

    logits_row = tl.load(
        logits_indices_ptr + row,
        mask=is_target_row,
        other=0,
    )
    draft_value = tl.load(
        logits_ptr + logits_row * logits_row_stride + safe_draft,
        mask=valid_draft,
        other=float("-inf"),
    ).to(tl.float32)
    penalty = tl.load(
        repetition_penalties_ptr + request_index,
        mask=is_target_row,
        other=1.0,
    )
    penalized = draft_value * tl.where(
        draft_value > 0,
        1.0 / penalty,
        penalty,
    )
    draft_value = tl.where(seen, penalized, draft_value)
    tl.store(
        relaxed_mask_ptr + row,
        valid_draft & ((max_value - draft_value) <= max_margin),
        mask=is_target_row,
    )


class _FusedGreedyScratch:
    """Reusable buffers for the exact repetition-aware argmax path."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.partial_values: torch.Tensor | None = None
        self.partial_ids: torch.Tensor | None = None
        self.output_ids: torch.Tensor | None = None
        self.relaxed_mask: torch.Tensor | None = None

    def ensure(
        self,
        *,
        num_rows: int,
        num_partials: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.partial_values is None
            or self.partial_values.shape[0] < num_rows
            or self.partial_values.shape[1] < num_partials
        ):
            shape = (num_rows, num_partials)
            self.partial_values = torch.empty(
                shape,
                dtype=torch.float32,
                device=self.device,
            )
            self.partial_ids = torch.empty(
                shape,
                dtype=torch.int32,
                device=self.device,
            )
        if self.output_ids is None or self.output_ids.shape[0] < num_rows:
            self.output_ids = torch.empty(
                num_rows,
                dtype=torch.int64,
                device=self.device,
            )
        assert self.partial_ids is not None
        return self.partial_values, self.partial_ids, self.output_ids

    def ensure_relaxed_mask(self, num_rows: int) -> torch.Tensor:
        if self.relaxed_mask is None or self.relaxed_mask.shape[0] < num_rows:
            self.relaxed_mask = torch.empty(
                num_rows,
                dtype=torch.bool,
                device=self.device,
            )
        return self.relaxed_mask


_FUSED_GREEDY_SCRATCH: dict[str, _FusedGreedyScratch] = {}


@triton.jit
def _apply_sparse_repetition_kernel(
    logits_ptr,
    history_ptr,
    cu_num_draft_tokens_ptr,
    spec_token_ids_ptr,
    repetition_penalties_ptr,
    logits_row_stride,
    history_row_stride,
    spec_row_stride,
    num_requests,
    total_draft_tokens,
    vocab_size,
    REQUEST_BLOCK_SIZE: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
    HISTORY_BLOCK_SIZE: tl.constexpr,
):
    """Apply repetition scaling only at unique token IDs seen by each row."""
    row = tl.program_id(0)
    is_target = row < total_draft_tokens
    request_offsets = tl.arange(0, REQUEST_BLOCK_SIZE)
    cumulative = tl.load(
        cu_num_draft_tokens_ptr + request_offsets,
        mask=request_offsets < num_requests,
        other=total_draft_tokens,
    )
    target_request = tl.sum(
        (is_target & (row >= cumulative)).to(tl.int32),
        axis=0,
    )
    request_index = tl.where(
        is_target,
        target_request,
        row - total_draft_tokens,
    )
    request_start = tl.load(
        cu_num_draft_tokens_ptr + request_index - 1,
        mask=is_target & (request_index > 0),
        other=0,
    )
    local_position = row - request_start
    penalty = tl.load(repetition_penalties_ptr + request_index)

    history_offsets = tl.arange(0, HISTORY_BLOCK_SIZE)
    history_ids = tl.load(
        history_ptr + request_index * history_row_stride + history_offsets,
    )
    valid_history = (history_ids >= 0) & (history_ids < vocab_size)
    safe_history_ids = tl.where(valid_history, history_ids, 0)
    values = tl.load(
        logits_ptr + row * logits_row_stride + safe_history_ids,
        mask=valid_history,
        other=0.0,
    )
    values *= tl.where(values > 0, 1.0 / penalty, penalty)
    tl.store(
        logits_ptr + row * logits_row_stride + safe_history_ids,
        values,
        mask=valid_history,
    )

    for position in tl.static_range(0, MAX_SPEC_LEN):
        spec_token = tl.load(
            spec_token_ids_ptr + request_index * spec_row_stride + position,
        )
        already_seen = (
            tl.sum(
                (valid_history & (history_ids == spec_token)).to(tl.int32),
                axis=0,
            )
            > 0
        )
        for earlier_position in tl.static_range(0, position):
            already_seen |= (
                tl.load(
                    spec_token_ids_ptr + request_index * spec_row_stride + earlier_position,
                )
                == spec_token
            )
        apply_prefix = (
            is_target
            & (position < local_position)
            & ~already_seen
            & (spec_token >= 0)
            & (spec_token < vocab_size)
        )
        safe_spec_token = tl.where(apply_prefix, spec_token, 0)
        value = tl.load(
            logits_ptr + row * logits_row_stride + safe_spec_token,
            mask=apply_prefix,
            other=0.0,
        )
        value *= tl.where(value > 0, 1.0 / penalty, penalty)
        tl.store(
            logits_ptr + row * logits_row_stride + safe_spec_token,
            value,
            mask=apply_prefix,
        )


@triton.jit
def _mask_seen_and_reduce_repetition_kernel(
    logits_ptr,
    history_ptr,
    logits_indices_ptr,
    repeat_indices_ptr,
    local_positions_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    repetition_penalties_ptr,
    seen_best_values_ptr,
    seen_best_ids_ptr,
    logits_row_stride,
    history_row_stride,
    num_rows,
    vocab_size,
    HISTORY_BLOCK_SIZE: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
):
    """Mask sparse seen IDs and reduce their repetition-adjusted maximum."""
    output_row = tl.program_id(0)
    request_index = tl.load(repeat_indices_ptr + output_row)
    logits_row = tl.load(logits_indices_ptr + output_row)
    local_position = tl.load(local_positions_ptr + output_row)
    request_start = tl.load(
        cu_num_draft_tokens_ptr + request_index - 1,
        mask=(request_index > 0) & (local_position > 0),
        other=0,
    )
    penalty = tl.load(repetition_penalties_ptr + request_index)

    history_offsets = tl.arange(0, HISTORY_BLOCK_SIZE)
    history_ids = tl.load(
        history_ptr + request_index * history_row_stride + history_offsets,
    )
    valid_history = (history_ids >= 0) & (history_ids < vocab_size)
    safe_history_ids = tl.where(valid_history, history_ids, 0)
    history_values = tl.load(
        logits_ptr + logits_row * logits_row_stride + safe_history_ids,
        mask=valid_history,
        other=float("-inf"),
    ).to(tl.float32)
    history_values = history_values * tl.where(
        history_values > 0,
        1.0 / penalty,
        penalty,
    )
    best_value = tl.max(history_values, axis=0)
    best_token = tl.min(
        tl.where(history_values == best_value, history_ids, vocab_size),
        axis=0,
    )

    for position in tl.static_range(0, MAX_SPEC_LEN):
        prefix_id = tl.load(
            draft_token_ids_ptr + request_start + position,
            mask=position < local_position,
            other=vocab_size,
        ).to(tl.int64)
        valid_prefix = (
            (output_row < num_rows)
            & (position < local_position)
            & (prefix_id >= 0)
            & (prefix_id < vocab_size)
        )
        safe_prefix_id = tl.where(valid_prefix, prefix_id, 0)
        prefix_value = tl.load(
            logits_ptr + logits_row * logits_row_stride + safe_prefix_id,
            mask=valid_prefix,
            other=float("-inf"),
        ).to(tl.float32)
        prefix_value = prefix_value * tl.where(
            prefix_value > 0,
            1.0 / penalty,
            penalty,
        )
        better = (prefix_value > best_value) | (
            (prefix_value == best_value) & (prefix_id < best_token)
        )
        best_value = tl.where(better, prefix_value, best_value)
        best_token = tl.where(better, prefix_id, best_token)

    tl.store(
        logits_ptr + logits_row * logits_row_stride + safe_history_ids,
        float("-inf"),
        mask=valid_history,
    )
    for position in tl.static_range(0, MAX_SPEC_LEN):
        prefix_id = tl.load(
            draft_token_ids_ptr + request_start + position,
            mask=position < local_position,
            other=vocab_size,
        )
        valid_prefix = (
            (output_row < num_rows)
            & (position < local_position)
            & (prefix_id >= 0)
            & (prefix_id < vocab_size)
        )
        tl.store(
            logits_ptr + logits_row * logits_row_stride + tl.where(valid_prefix, prefix_id, 0),
            float("-inf"),
            mask=valid_prefix,
        )

    tl.store(seen_best_values_ptr + output_row, best_value)
    tl.store(seen_best_ids_ptr + output_row, best_token)


class _SparseRepetitionState:
    """Incrementally materialize unique committed histories by request ID."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        history_width: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.history_width = history_width
        self.vocab_size = vocab_size
        self.history_cpu = torch.full(
            (max_batch_size, history_width),
            vocab_size,
            dtype=torch.int64,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self.history = torch.empty_like(self.history_cpu, device=device)
        self.row_req_ids: list[str | None] = [None] * max_batch_size
        self.requests: dict[str, dict[str, Any]] = {}
        self.layout_cache: dict[
            tuple[int, ...],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _new_request(
        self,
        prompt_tokens: Any,
        prompt_length: int,
        output_tokens: tuple[int, ...],
    ) -> dict[str, Any]:
        prompt = tuple(
            int(token_id)
            for token_id in prompt_tokens[:prompt_length]
            if 0 <= int(token_id) < self.vocab_size
        )
        unique_tokens = list(
            dict.fromkeys(
                (*prompt, *(token for token in output_tokens if 0 <= token < self.vocab_size))
            )
        )
        return {
            "prompt": prompt,
            "tokens": unique_tokens,
            "seen": set(unique_tokens),
            "output": output_tokens,
        }

    def update(self, sampling_metadata: Any, num_requests: int) -> bool:
        req_ids = getattr(sampling_metadata, "_vspec_req_ids", None)
        prompt_token_ids = getattr(
            sampling_metadata,
            "_vspec_prompt_token_ids_cpu",
            None,
        )
        prompt_lengths = getattr(
            sampling_metadata,
            "_vspec_prompt_lengths_cpu",
            None,
        )
        if (
            req_ids is None
            or prompt_token_ids is None
            or prompt_lengths is None
            or num_requests > self.max_batch_size
            or len(req_ids) < num_requests
            or len(sampling_metadata.output_token_ids) < num_requests
        ):
            return False

        active_req_ids: set[str] = set()
        for row in range(num_requests):
            req_id = req_ids[row]
            if req_id is None:
                return False
            active_req_ids.add(req_id)
            output_tokens = tuple(
                int(token_id) for token_id in sampling_metadata.output_token_ids[row]
            )
            request = self.requests.get(req_id)
            rebuilt = request is None
            appended_tokens: list[int] = []
            if request is None:
                request = self._new_request(
                    prompt_token_ids[row],
                    int(prompt_lengths[row]),
                    output_tokens,
                )
                self.requests[req_id] = request
            elif (
                len(output_tokens) >= len(request["output"])
                and output_tokens[: len(request["output"])] == request["output"]
            ):
                for token_id in output_tokens[len(request["output"]) :]:
                    if 0 <= token_id < self.vocab_size and token_id not in request["seen"]:
                        request["seen"].add(token_id)
                        request["tokens"].append(token_id)
                        appended_tokens.append(token_id)
                request["output"] = output_tokens
            elif output_tokens != request["output"]:
                request = self._new_request(
                    request["prompt"],
                    len(request["prompt"]),
                    output_tokens,
                )
                self.requests[req_id] = request
                rebuilt = True

            if len(request["tokens"]) > self.history_width:
                return False
            if rebuilt or self.row_req_ids[row] != req_id:
                self.history_cpu[row].fill_(self.vocab_size)
                token_count = len(request["tokens"])
                if token_count:
                    self.history_cpu[row, :token_count] = torch.as_tensor(
                        request["tokens"],
                        dtype=torch.int64,
                    )
            elif appended_tokens:
                start = len(request["tokens"]) - len(appended_tokens)
                self.history_cpu[row, start : start + len(appended_tokens)] = torch.as_tensor(
                    appended_tokens, dtype=torch.int64
                )
            self.row_req_ids[row] = req_id

        for row in range(num_requests, self.max_batch_size):
            self.row_req_ids[row] = None
        if len(self.requests) > self.max_batch_size * 4:
            self.requests = {req_id: self.requests[req_id] for req_id in active_req_ids}
        self.history[:num_requests].copy_(
            self.history_cpu[:num_requests],
            non_blocking=True,
        )
        return True

    def row_layout(
        self,
        num_draft_tokens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(int(length) for length in num_draft_tokens)
        cached = self.layout_cache.get(key)
        if cached is not None:
            return cached
        repeat_indices, local_positions = _row_layout(num_draft_tokens)
        cached = (
            torch.tensor(repeat_indices, dtype=torch.int64, device=self.history.device),
            torch.tensor(local_positions, dtype=torch.int64, device=self.history.device),
        )
        if len(self.layout_cache) >= 32:
            self.layout_cache.clear()
        self.layout_cache[key] = cached
        return cached


class _NativeRepetitionScratch:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: torch.Tensor | None = None
        self.ids: torch.Tensor | None = None

    def ensure(self, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.values is None or self.values.shape[0] < num_rows:
            self.values = torch.empty(num_rows, dtype=torch.float32, device=self.device)
            self.ids = torch.empty(num_rows, dtype=torch.int64, device=self.device)
        assert self.ids is not None
        return self.values[:num_rows], self.ids[:num_rows]


def native_sparse_repetition_greedy(
    owner: Any,
    metadata: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use native argmax after sparsely removing repetition-seen token IDs."""
    prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
    if prompt_token_ids is None or logits.ndim != 2:
        return None
    num_requests = len(metadata.num_draft_tokens)
    target_rows = int(metadata.target_logits_indices.numel())
    if sum(metadata.num_draft_tokens) != target_rows:
        return None

    vocab_size = int(logits.shape[-1])
    max_batch_size = int(getattr(sampling_metadata, "_vspec_max_num_reqs", num_requests))
    history_width = int(os.environ.get("VSPEC_DRAFT_NATIVE_REPETITION_HISTORY_WIDTH", "512"))
    if history_width <= 0 or history_width & (history_width - 1):
        raise ValueError("native repetition history width must be a positive power of two")
    state = getattr(owner, "_vspec_native_repetition_state", None)
    if (
        state is None
        or state.vocab_size != vocab_size
        or state.max_batch_size < max_batch_size
        or state.history_width != history_width
    ):
        state = _SparseRepetitionState(
            max_batch_size=max_batch_size,
            history_width=history_width,
            vocab_size=vocab_size,
            device=logits.device,
        )
        owner._vspec_native_repetition_state = state
    if not state.update(sampling_metadata, num_requests):
        return None

    row_indices = torch.cat((metadata.target_logits_indices, metadata.bonus_logits_indices))
    repeat_indices, local_positions = state.row_layout(metadata.num_draft_tokens)
    num_rows = int(row_indices.numel())
    scratch = getattr(owner, "_vspec_native_repetition_scratch", None)
    if scratch is None:
        scratch = _NativeRepetitionScratch(logits.device)
        owner._vspec_native_repetition_scratch = scratch
    seen_values, seen_ids = scratch.ensure(num_rows)

    _mask_seen_and_reduce_repetition_kernel[(num_rows,)](
        logits,
        state.history,
        row_indices,
        repeat_indices,
        local_positions,
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        sampling_metadata.repetition_penalties,
        seen_values,
        seen_ids,
        logits.stride(0),
        state.history.stride(0),
        num_rows,
        vocab_size,
        HISTORY_BLOCK_SIZE=history_width,
        MAX_SPEC_LEN=int(metadata.max_spec_len),
    )

    unseen_ids = logits.argmax(dim=-1)[row_indices]
    unseen_values = logits[row_indices, unseen_ids].to(torch.float32)
    choose_seen = (seen_values > unseen_values) | (
        (seen_values == unseen_values) & (seen_ids < unseen_ids)
    )
    token_ids = torch.where(choose_seen, seen_ids, unseen_ids)
    return (
        token_ids[:target_rows],
        token_ids[target_rows:].to(torch.int32).unsqueeze(-1),
    )


class _PackedRepetitionState:
    """Incremental request history encoded as one bit per vocabulary ID."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        vocab_size: int,
        device: torch.device,
        track_token_lists: bool = False,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.seen_mask = torch.zeros(
            (max_batch_size, vocab_size),
            dtype=torch.bool,
            device=device,
        )
        self.track_token_lists = track_token_lists
        self.seen_tokens = (
            torch.empty(
                (max_batch_size, vocab_size),
                dtype=torch.int32,
                device=device,
            )
            if track_token_lists
            else None
        )
        self.seen_counts_cpu = (
            torch.zeros(
                max_batch_size,
                dtype=torch.int32,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
            if track_token_lists
            else None
        )
        self.seen_counts = (
            torch.zeros(
                max_batch_size,
                dtype=torch.int32,
                device=device,
            )
            if track_token_lists
            else None
        )
        self.active_slots_cpu = torch.empty(
            max_batch_size,
            dtype=torch.int64,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        self.active_slots = torch.empty(
            max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self.requests: dict[str, dict[str, Any]] = {}
        self.free_slots = list(range(max_batch_size - 1, -1, -1))
        self.update_capacity = 0
        self.update_slots_cpu: torch.Tensor | None = None
        self.update_tokens_cpu: torch.Tensor | None = None
        self.update_positions_cpu: torch.Tensor | None = None
        self.update_slots: torch.Tensor | None = None
        self.update_tokens: torch.Tensor | None = None
        self.update_positions: torch.Tensor | None = None
        self.layout_cache: dict[
            tuple[int, ...],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    @staticmethod
    def _tokens_from_prompt(
        prompt_token_ids: Any,
        prompt_length: int,
    ) -> list[int]:
        prompt = prompt_token_ids[:prompt_length]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        return [int(token_id) for token_id in prompt]

    def _append_updates(
        self,
        slot_indices: list[int],
        token_ids: list[int],
        history_positions: list[int] | None = None,
    ) -> None:
        if not token_ids:
            return
        update_count = len(token_ids)
        if update_count > self.update_capacity:
            capacity = 1 << (update_count - 1).bit_length()
            pinned = is_pin_memory_available()
            self.update_slots_cpu = torch.empty(
                capacity,
                dtype=torch.int64,
                device="cpu",
                pin_memory=pinned,
            )
            self.update_tokens_cpu = torch.empty(
                capacity,
                dtype=torch.int64,
                device="cpu",
                pin_memory=pinned,
            )
            if self.track_token_lists:
                self.update_positions_cpu = torch.empty(
                    capacity,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=pinned,
                )
            self.update_slots = torch.empty(
                capacity,
                dtype=torch.int64,
                device=self.device,
            )
            self.update_tokens = torch.empty(
                capacity,
                dtype=torch.int64,
                device=self.device,
            )
            if self.track_token_lists:
                self.update_positions = torch.empty(
                    capacity,
                    dtype=torch.int64,
                    device=self.device,
                )
            self.update_capacity = capacity
        assert self.update_slots_cpu is not None
        assert self.update_tokens_cpu is not None
        assert self.update_slots is not None
        assert self.update_tokens is not None
        self.update_slots_cpu[:update_count].copy_(torch.as_tensor(slot_indices, dtype=torch.int64))
        self.update_tokens_cpu[:update_count].copy_(torch.as_tensor(token_ids, dtype=torch.int64))
        slots = self.update_slots[:update_count]
        tokens = self.update_tokens[:update_count]
        slots.copy_(self.update_slots_cpu[:update_count], non_blocking=True)
        tokens.copy_(self.update_tokens_cpu[:update_count], non_blocking=True)
        block_size = 256
        if self.track_token_lists:
            if history_positions is None:
                raise RuntimeError("tracked repetition history requires positions")
            assert self.update_positions_cpu is not None
            assert self.update_positions is not None
            assert self.seen_tokens is not None
            self.update_positions_cpu[:update_count].copy_(
                torch.as_tensor(history_positions, dtype=torch.int64)
            )
            positions = self.update_positions[:update_count]
            positions.copy_(
                self.update_positions_cpu[:update_count],
                non_blocking=True,
            )
            _append_seen_tokens_and_history_kernel[
                (triton.cdiv(update_count, block_size),)
            ](
                self.seen_mask,
                self.seen_tokens,
                slots,
                tokens,
                positions,
                self.seen_mask.stride(0),
                self.seen_tokens.stride(0),
                update_count,
                self.vocab_size,
                BLOCK_SIZE=block_size,
            )
        else:
            _append_seen_tokens_kernel[(triton.cdiv(update_count, block_size),)](
                self.seen_mask,
                slots,
                tokens,
                self.seen_mask.stride(0),
                update_count,
                self.vocab_size,
                BLOCK_SIZE=block_size,
            )

    def update(
        self,
        sampling_metadata: Any,
        num_requests: int,
    ) -> torch.Tensor | None:
        req_ids = getattr(sampling_metadata, "_vspec_req_ids", None)
        prompt_token_ids = getattr(
            sampling_metadata,
            "_vspec_prompt_token_ids_cpu",
            None,
        )
        prompt_lengths = getattr(
            sampling_metadata,
            "_vspec_prompt_lengths_cpu",
            None,
        )
        if (
            req_ids is None
            or prompt_token_ids is None
            or prompt_lengths is None
            or num_requests > self.max_batch_size
            or len(req_ids) < num_requests
            or len(sampling_metadata.output_token_ids) < num_requests
        ):
            return None

        active_ids = set(req_ids[:num_requests])
        for req_id in tuple(self.requests):
            if req_id not in active_ids:
                self.free_slots.append(self.requests.pop(req_id)["slot"])

        update_slots: list[int] = []
        update_tokens: list[int] = []
        update_positions: list[int] = []
        for row, req_id in enumerate(req_ids[:num_requests]):
            output = sampling_metadata.output_token_ids[row]
            request = self.requests.get(req_id)
            rebuild = request is None
            if request is None:
                if not self.free_slots:
                    return None
                request = {
                    "slot": self.free_slots.pop(),
                    "output_len": 0,
                    "last_output_token": None,
                    "seen": set(),
                }
                self.requests[req_id] = request
            else:
                old_length = int(request["output_len"])
                old_last_token = request["last_output_token"]
                if len(output) < old_length or (
                    old_length > 0 and int(output[old_length - 1]) != old_last_token
                ):
                    rebuild = True

            old_length = int(request["output_len"])
            if rebuild:
                old_length = 0

            slot = int(request["slot"])
            seen = request["seen"]
            if rebuild:
                self.seen_mask[slot].zero_()
                seen.clear()
                if self.seen_counts_cpu is not None:
                    self.seen_counts_cpu[slot] = 0
                tokens = self._tokens_from_prompt(
                    prompt_token_ids[row],
                    int(prompt_lengths[row]),
                )
                tokens.extend(output)
            else:
                tokens = output[old_length:]
            for token_id in tokens:
                token_id = int(token_id)
                if 0 <= token_id < self.vocab_size and token_id not in seen:
                    update_positions.append(len(seen))
                    seen.add(token_id)
                    update_slots.append(slot)
                    update_tokens.append(token_id)
            if self.seen_counts_cpu is not None:
                self.seen_counts_cpu[slot] = len(seen)
            request["output_len"] = len(output)
            request["last_output_token"] = int(output[-1]) if output else None
            self.active_slots_cpu[row] = slot

        self._append_updates(
            update_slots,
            update_tokens,
            update_positions if self.track_token_lists else None,
        )
        if self.seen_counts is not None:
            assert self.seen_counts_cpu is not None
            self.seen_counts.copy_(self.seen_counts_cpu, non_blocking=True)
        self.active_slots[:num_requests].copy_(
            self.active_slots_cpu[:num_requests],
            non_blocking=True,
        )
        return self.active_slots[:num_requests]

    def row_layout(
        self,
        num_draft_tokens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(int(length) for length in num_draft_tokens)
        cached = self.layout_cache.get(key)
        if cached is not None:
            return cached
        repeat_indices, local_positions = _row_layout(num_draft_tokens)
        cached = (
            torch.tensor(repeat_indices, dtype=torch.int64, device=self.device),
            torch.tensor(local_positions, dtype=torch.int64, device=self.device),
        )
        if len(self.layout_cache) >= 32:
            self.layout_cache.clear()
        self.layout_cache[key] = cached
        return cached


class _DraftFusedRepetitionState:
    """Persistent inputs and scratch buffers for Draft-side greedy sampling."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.packed = _PackedRepetitionState(
            max_batch_size=max_batch_size,
            vocab_size=vocab_size,
            device=device,
        )
        self.row_indices = torch.arange(
            max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self.local_positions = (
            torch.zeros(max_batch_size, dtype=torch.int64, device=device),
            torch.ones(max_batch_size, dtype=torch.int64, device=device),
        )
        self.cu_one_token = torch.arange(
            1,
            max_batch_size + 1,
            dtype=torch.int64,
            device=device,
        )
        self.dummy_prefix = torch.zeros(
            max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self.scratches = (
            _FusedGreedyScratch(device),
            _FusedGreedyScratch(device),
        )
        self.request_slots: torch.Tensor | None = None


def fused_draft_repetition_greedy(
    owner: Any,
    logits: Any,
    sampling_metadata: Any,
    prefix_token_ids: list[torch.Tensor],
) -> torch.Tensor | None:
    """Apply exact repetition-aware argmax directly to Draft logits."""
    logits_tensor = getattr(logits, "logits", logits)
    active_ids = getattr(logits, "active_ids", None)
    if (
        not isinstance(logits_tensor, torch.Tensor)
        or logits_tensor.ndim != 2
        or not sampling_metadata.all_greedy
        or not getattr(sampling_metadata, "_vspec_repetition_only", False)
        or len(prefix_token_ids) > 1
    ):
        return None

    num_rows, vocab_size = logits_tensor.shape
    if num_rows == 0:
        return None
    token_id_limit = int(getattr(owner, "_vspec_draft_full_vocab_size", vocab_size))
    max_batch_size = int(getattr(sampling_metadata, "_vspec_max_num_reqs", num_rows))
    state = getattr(owner, "_vspec_draft_fused_repetition_state", None)
    if state is None or state.vocab_size != token_id_limit or state.max_batch_size < max_batch_size:
        state = _DraftFusedRepetitionState(
            max_batch_size=max_batch_size,
            vocab_size=token_id_limit,
            device=logits_tensor.device,
        )
        owner._vspec_draft_fused_repetition_state = state

    step = len(prefix_token_ids)
    if step == 0:
        state.request_slots = state.packed.update(
            sampling_metadata,
            num_rows,
        )
    request_slots = state.request_slots
    if request_slots is None or request_slots.shape[0] < num_rows:
        return None

    vocab_block_size = 2048
    num_vocab_blocks = triton.cdiv(vocab_size, vocab_block_size)
    num_partials = min(16, num_vocab_blocks)
    partial_values, partial_ids, token_ids = state.scratches[step].ensure(
        num_rows=num_rows,
        num_partials=num_partials,
    )
    partial_values = partial_values[:num_rows, :num_partials]
    partial_ids = partial_ids[:num_rows, :num_partials]
    token_ids = token_ids[:num_rows]
    prefix = prefix_token_ids[0] if prefix_token_ids else state.dummy_prefix[:num_rows]

    _repetition_argmax_partials_kernel[(num_rows, num_partials)](
        logits_tensor,
        active_ids if active_ids is not None else state.row_indices,
        state.packed.seen_mask,
        state.row_indices,
        request_slots,
        state.row_indices,
        state.local_positions[step],
        state.cu_one_token,
        prefix,
        sampling_metadata.repetition_penalties,
        partial_values,
        partial_ids,
        logits_tensor.stride(0),
        state.packed.seen_mask.stride(0),
        partial_values.stride(0),
        vocab_size,
        token_id_limit,
        MAX_SPEC_LEN=1,
        USE_VOCAB_IDS=active_ids is not None,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        VOCAB_GRID_SIZE=num_partials,
        VOCAB_BLOCK_SIZE=vocab_block_size,
    )
    _repetition_argmax_finalize_kernel[(num_rows,)](
        partial_values,
        partial_ids,
        token_ids,
        partial_values.stride(0),
        num_partials,
        PARTIAL_BLOCK_SIZE=triton.next_power_of_2(num_partials),
    )
    return token_ids


class _DraftSparseProposalState:
    """Incremental sparse history used by exact Draft-side sampling."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.packed = _PackedRepetitionState(
            max_batch_size=max_batch_size,
            vocab_size=vocab_size,
            device=device,
            track_token_lists=True,
        )
        self.request_slots: torch.Tensor | None = None


def apply_sparse_draft_repetition_penalty(
    owner: Any,
    logits: Any,
    sampling_metadata: Any,
    prefix_token_ids: list[torch.Tensor],
) -> Any | None:
    """Apply exact repetition penalty only to token IDs present in history."""
    logits_tensor = getattr(logits, "logits", logits)
    active_ids = getattr(logits, "active_ids", None)
    if (
        not isinstance(logits_tensor, torch.Tensor)
        or logits_tensor.ndim != 2
        or active_ids is not None
        or not sampling_metadata.all_greedy
        or not getattr(sampling_metadata, "_vspec_repetition_only", False)
        or len(prefix_token_ids) > 1
    ):
        return None

    num_rows, vocab_size = logits_tensor.shape
    if num_rows == 0:
        return None
    max_batch_size = int(
        getattr(sampling_metadata, "_vspec_max_num_reqs", num_rows)
    )
    state = getattr(owner, "_vspec_draft_sparse_repetition_state", None)
    if (
        state is None
        or state.vocab_size != vocab_size
        or state.max_batch_size < max_batch_size
    ):
        state = _DraftSparseProposalState(
            max_batch_size=max_batch_size,
            vocab_size=vocab_size,
            device=logits_tensor.device,
        )
        owner._vspec_draft_sparse_repetition_state = state

    if not prefix_token_ids:
        state.request_slots = state.packed.update(
            sampling_metadata,
            num_rows,
        )
    request_slots = state.request_slots
    if request_slots is None or request_slots.shape[0] < num_rows:
        return None

    history_width = max(
        (
            len(request["seen"])
            for request in state.packed.requests.values()
        ),
        default=0,
    )
    if history_width:
        assert state.packed.seen_tokens is not None
        assert state.packed.seen_counts is not None
        history_block = 256
        history_blocks = triton.cdiv(history_width, history_block)
        _apply_sparse_history_repetition_kernel[(num_rows, history_blocks)](
            logits_tensor,
            state.packed.seen_tokens,
            state.packed.seen_counts,
            request_slots,
            sampling_metadata.repetition_penalties,
            logits_tensor.stride(0),
            state.packed.seen_tokens.stride(0),
            num_rows,
            vocab_size,
            HISTORY_BLOCK=history_block,
        )

    if prefix_token_ids:
        _apply_sparse_prefix_repetition_kernel[(num_rows,)](
            logits_tensor,
            prefix_token_ids[0],
            state.packed.seen_mask,
            request_slots,
            sampling_metadata.repetition_penalties,
            logits_tensor.stride(0),
            state.packed.seen_mask.stride(0),
            num_rows,
            vocab_size,
        )
    return logits


def _row_layout(
    num_draft_tokens: list[int],
) -> tuple[list[int], list[int]]:
    repeat_indices = [
        request_index
        for request_index, length in enumerate(num_draft_tokens)
        for _ in range(length)
    ]
    local_positions = [position for length in num_draft_tokens for position in range(length)]
    repeat_indices.extend(range(len(num_draft_tokens)))
    local_positions.extend(num_draft_tokens)
    return repeat_indices, local_positions


def select_exact_repetition_topk(
    candidate_values: torch.Tensor,
    candidate_ids: torch.Tensor,
    seen_candidates: torch.Tensor,
    repetition_penalties: torch.Tensor,
    *,
    vocab_size: int,
    has_outside_candidates: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exact repetition-aware winners when Top-K proves optimality."""
    if candidate_values.ndim != 2 or candidate_ids.shape != candidate_values.shape:
        raise ValueError("Top-K values and IDs must be matching two-dimensional tensors")
    if seen_candidates.shape != candidate_values.shape:
        raise ValueError("Top-K seen mask must match the candidate tensors")
    if candidate_values.shape[1] == 0:
        raise ValueError("Top-K candidate tensors must not be empty")

    raw_values = candidate_values.to(torch.float32)
    penalties = repetition_penalties.to(torch.float32).reshape(-1, 1)
    if penalties.shape[0] != raw_values.shape[0]:
        raise ValueError("one repetition penalty is required per candidate row")
    penalized_values = torch.where(
        raw_values > 0,
        raw_values / penalties,
        raw_values * penalties,
    )
    adjusted_values = torch.where(
        seen_candidates,
        penalized_values,
        raw_values,
    )
    best_values = adjusted_values.max(dim=-1, keepdim=True).values

    # Match torch.argmax's lowest-token-ID tie break without depending on the
    # order returned by topk(). A boundary tie cannot be proven because Top-K
    # may have omitted a lower token ID, so it falls back below.
    sentinel = torch.full_like(candidate_ids, vocab_size)
    winner_ids = (
        torch.where(
            adjusted_values == best_values,
            candidate_ids,
            sentinel,
        )
        .min(dim=-1)
        .values
    )
    if has_outside_candidates:
        proven = best_values.squeeze(-1) > raw_values[:, -1]
    else:
        proven = torch.ones_like(winner_ids, dtype=torch.bool)
    return winner_ids, proven


def exact_topk_repetition_greedy(
    owner: Any,
    metadata: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    candidate_count: int,
    *,
    synchronize_proof: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply exact greedy repetition penalty to a provably sufficient Top-K."""
    if candidate_count <= 0:
        return None
    prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
    if prompt_token_ids is None:
        return None
    num_requests = len(metadata.num_draft_tokens)
    target_rows = int(metadata.target_logits_indices.numel())
    if (
        num_requests == 0
        or sum(metadata.num_draft_tokens) != target_rows
        or metadata.draft_token_ids.ndim != 1
    ):
        return None

    vocab_size = logits.shape[-1]
    max_batch_size = int(getattr(sampling_metadata, "_vspec_max_num_reqs", num_requests))
    state = getattr(owner, "_vspec_topk_repetition_state", None)
    if state is None or state.vocab_size != vocab_size or state.max_batch_size < max_batch_size:
        state = _PackedRepetitionState(
            max_batch_size=max_batch_size,
            vocab_size=vocab_size,
            device=logits.device,
        )
        owner._vspec_topk_repetition_state = state
    request_slots = state.update(sampling_metadata, num_requests)
    if request_slots is None:
        return None

    row_indices = torch.cat((metadata.target_logits_indices, metadata.bonus_logits_indices))
    candidate_count = min(candidate_count, vocab_size)
    selected_logits = torch.index_select(logits, 0, row_indices)
    candidate_values, candidate_ids = selected_logits.topk(
        candidate_count,
        dim=-1,
        sorted=True,
    )

    repeat_indices, local_positions = state.row_layout(metadata.num_draft_tokens)
    row_slots = request_slots[repeat_indices]
    seen_candidates = state.seen_mask[
        row_slots.unsqueeze(-1),
        candidate_ids,
    ]

    max_spec_len = int(metadata.max_spec_len)
    if max_spec_len:
        cu_num_draft_tokens = metadata.cu_num_draft_tokens
        request_starts = torch.cat(
            (
                torch.zeros_like(cu_num_draft_tokens[:1]),
                cu_num_draft_tokens[:-1],
            )
        )
        prefix_offsets = torch.arange(
            max_spec_len,
            device=logits.device,
            dtype=request_starts.dtype,
        )
        prefix_indices = request_starts[repeat_indices].unsqueeze(-1) + prefix_offsets.unsqueeze(0)
        valid_prefix = prefix_offsets.unsqueeze(0) < local_positions.unsqueeze(-1)
        safe_prefix_indices = prefix_indices.clamp_max(max(target_rows - 1, 0))
        prefix_ids = metadata.draft_token_ids[safe_prefix_indices]
        seen_candidates |= (
            (candidate_ids.unsqueeze(-1) == prefix_ids.unsqueeze(1)) & valid_prefix.unsqueeze(1)
        ).any(dim=-1)

    row_penalties = sampling_metadata.repetition_penalties[repeat_indices]
    winner_ids, proven = select_exact_repetition_topk(
        candidate_values,
        candidate_ids,
        seen_candidates,
        row_penalties,
        vocab_size=vocab_size,
        has_outside_candidates=candidate_count < vocab_size,
    )
    is_proven = not synchronize_proof or bool(proven.all().item())
    if not is_proven:
        owner._vspec_topk_repetition_fallbacks = (
            getattr(owner, "_vspec_topk_repetition_fallbacks", 0) + 1
        )
        _trace_exact_topk_repetition(owner, candidate_count)
        return None

    owner._vspec_topk_repetition_hits = getattr(owner, "_vspec_topk_repetition_hits", 0) + 1
    _trace_exact_topk_repetition(owner, candidate_count)
    return (
        winner_ids[:target_rows],
        winner_ids[target_rows:].to(torch.int32).unsqueeze(-1),
    )


def _trace_exact_topk_repetition(owner: Any, candidate_count: int) -> None:
    trace_enabled = os.environ.get(
        "HUST_VSPEC_DRAFT_EXACT_REPETITION_TRACE",
        os.environ.get("VSPEC_DRAFT_EXACT_REPETITION_TRACE", "0"),
    )
    if trace_enabled != "1":
        return
    hits = int(getattr(owner, "_vspec_topk_repetition_hits", 0))
    fallbacks = int(getattr(owner, "_vspec_topk_repetition_fallbacks", 0))
    total = hits + fallbacks
    if total == 1 or total % 128 == 0:
        print(
            f"vSpec exact repetition Top-K: k={candidate_count} hits={hits} fallbacks={fallbacks}",
            flush=True,
        )


def fused_repetition_greedy(
    owner: Any,
    metadata: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    max_margin: float | None = None,
    *,
    active_ids: torch.Tensor | None = None,
    full_vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Select exact repetition-aware tokens without full-logit temporaries."""
    prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
    if prompt_token_ids is None:
        return None
    num_requests = len(metadata.num_draft_tokens)
    target_rows = int(metadata.target_logits_indices.numel())
    if sum(metadata.num_draft_tokens) != target_rows:
        return None

    compact_vocab_size = logits.shape[-1]
    token_id_limit = compact_vocab_size if full_vocab_size is None else full_vocab_size
    if active_ids is not None:
        if active_ids.ndim != 1 or active_ids.numel() != compact_vocab_size:
            return None
        if max_margin is not None:
            # Margin lookup needs a full-token-ID to compact-column map. The
            # strict path used by Target active-vocabulary projection does not.
            return None
    max_batch_size = int(getattr(sampling_metadata, "_vspec_max_num_reqs", num_requests))
    state = getattr(owner, "_vspec_packed_repetition_state", None)
    if (
        state is None
        or state.vocab_size != token_id_limit
        or state.max_batch_size < max_batch_size
    ):
        state = _PackedRepetitionState(
            max_batch_size=max_batch_size,
            vocab_size=token_id_limit,
            device=logits.device,
        )
        owner._vspec_packed_repetition_state = state
    request_slots = state.update(sampling_metadata, num_requests)
    if request_slots is None:
        return None

    row_indices = torch.cat((metadata.target_logits_indices, metadata.bonus_logits_indices))
    repeat_indices_tensor, local_positions_tensor = state.row_layout(metadata.num_draft_tokens)
    num_rows = int(row_indices.numel())
    vocab_block_size = int(
        os.environ.get("VSPEC_DRAFT_TARGET_REPETITION_BLOCK_SIZE", "2048")
    )
    if vocab_block_size not in {1024, 2048, 4096}:
        raise ValueError(
            "VSPEC_DRAFT_TARGET_REPETITION_BLOCK_SIZE must be one of "
            "1024, 2048, or 4096"
        )
    num_vocab_blocks = triton.cdiv(compact_vocab_size, vocab_block_size)
    partial_limit = int(
        os.environ.get("VSPEC_DRAFT_TARGET_REPETITION_PARTIALS", "16")
    )
    if partial_limit not in {8, 16, 32, 64}:
        raise ValueError(
            "VSPEC_DRAFT_TARGET_REPETITION_PARTIALS must be one of "
            "8, 16, 32, or 64"
        )
    num_partials = min(partial_limit, num_vocab_blocks)
    scratch_key = str(logits.device)
    scratch = _FUSED_GREEDY_SCRATCH.get(scratch_key)
    if scratch is None:
        scratch = _FusedGreedyScratch(logits.device)
        _FUSED_GREEDY_SCRATCH[scratch_key] = scratch
    partial_values, partial_ids, token_ids = scratch.ensure(
        num_rows=num_rows,
        num_partials=num_partials,
    )
    partial_values = partial_values[:num_rows, :num_partials]
    partial_ids = partial_ids[:num_rows, :num_partials]
    token_ids = token_ids[:num_rows]

    _repetition_argmax_partials_kernel[(num_rows, num_partials)](
        logits,
        active_ids if active_ids is not None else row_indices,
        state.seen_mask,
        row_indices,
        request_slots,
        repeat_indices_tensor,
        local_positions_tensor,
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        sampling_metadata.repetition_penalties,
        partial_values,
        partial_ids,
        logits.stride(0),
        state.seen_mask.stride(0),
        partial_values.stride(0),
        compact_vocab_size,
        token_id_limit,
        MAX_SPEC_LEN=int(metadata.max_spec_len),
        USE_VOCAB_IDS=active_ids is not None,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        VOCAB_GRID_SIZE=num_partials,
        VOCAB_BLOCK_SIZE=vocab_block_size,
    )
    partial_block_size = triton.next_power_of_2(num_partials)
    relaxed_mask = None
    if max_margin is None:
        _repetition_argmax_finalize_kernel[(num_rows,)](
            partial_values,
            partial_ids,
            token_ids,
            partial_values.stride(0),
            num_partials,
            PARTIAL_BLOCK_SIZE=partial_block_size,
        )
    else:
        relaxed_mask = scratch.ensure_relaxed_mask(target_rows)[:target_rows]
        _repetition_argmax_margin_finalize_kernel[(num_rows,)](
            partial_values,
            partial_ids,
            token_ids,
            relaxed_mask,
            logits,
            state.seen_mask,
            row_indices,
            request_slots,
            repeat_indices_tensor,
            local_positions_tensor,
            metadata.cu_num_draft_tokens,
            metadata.draft_token_ids,
            sampling_metadata.repetition_penalties,
            partial_values.stride(0),
            logits.stride(0),
            state.seen_mask.stride(0),
            num_partials,
            target_rows,
            compact_vocab_size,
            max_margin,
            PARTIAL_BLOCK_SIZE=partial_block_size,
            MAX_SPEC_LEN=int(metadata.max_spec_len),
        )
    return (
        token_ids[:target_rows],
        token_ids[target_rows:].to(torch.int32).unsqueeze(-1),
        relaxed_mask,
    )
