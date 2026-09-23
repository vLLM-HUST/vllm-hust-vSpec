"""Low-cost second-token proposal for serial Draft decoding."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available


@triton.jit
def _hybrid_second_token_kernel(
    history_ptr,
    lengths_ptr,
    base_token_ptr,
    first_token_ptr,
    transition_ptr,
    pair_keys_ptr,
    pair_values_ptr,
    output_ptr,
    history_row_stride,
    vocab_size,
    pair_table_mask,
    history_width: tl.constexpr,
    BLOCK: tl.constexpr,
    PAIR_PROBES: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * history_row_stride
    seq_len = tl.load(lengths_ptr + row)
    base_token = tl.load(base_token_ptr + row)
    first_token = tl.load(first_token_ptr + row)
    tl.store(history_ptr + row_offset + seq_len, base_token)
    tl.store(history_ptr + row_offset + seq_len + 1, first_token)
    total_len = seq_len + 2

    offsets = tl.arange(0, BLOCK)
    best_lengths = tl.zeros([BLOCK], dtype=tl.int32)
    for ngram_len in tl.static_range(1, 6):
        matches = tl.full([BLOCK], 1, dtype=tl.int1)
        for suffix_offset in tl.static_range(0, ngram_len):
            positions = offsets + suffix_offset
            values = tl.load(
                history_ptr + row_offset + positions,
                mask=positions < total_len,
                other=-1,
            )
            suffix_position = total_len - ngram_len + suffix_offset
            suffix_position = tl.where(suffix_position >= 0, suffix_position, 0)
            suffix = tl.load(history_ptr + row_offset + suffix_position)
            matches &= values == suffix
        valid = (total_len >= ngram_len) & (
            offsets < total_len - ngram_len
        )
        best_lengths = tl.where(
            matches & valid,
            ngram_len,
            best_lengths,
        )

    best_length = tl.max(best_lengths, axis=0)
    matching_best = (best_length > 0) & (best_lengths == best_length)
    best_position = tl.max(
        tl.where(matching_best, offsets, -1),
        axis=0,
    )
    candidate_position = best_position + best_length
    ngram_candidate = tl.load(
        history_ptr + row_offset + candidate_position,
        mask=best_length > 0,
        other=-1,
    )

    valid_first = (first_token >= 0) & (first_token < vocab_size)
    safe_first = tl.where(valid_first, first_token, 0)
    transition_candidate = tl.load(
        transition_ptr + safe_first,
        mask=valid_first,
        other=-1,
    )
    transition_candidate = tl.where(
        transition_candidate >= 0,
        transition_candidate,
        220,
    )
    pair_key = base_token.to(tl.int64) * vocab_size + first_token.to(tl.int64)
    pair_hash = (
        base_token.to(tl.int64) * 1315423911
        + first_token.to(tl.int64) * 2654435761
    ) & pair_table_mask
    pair_candidate = -1
    for probe in tl.static_range(0, PAIR_PROBES):
        pair_slot = (pair_hash + probe) & pair_table_mask
        stored_key = tl.load(pair_keys_ptr + pair_slot)
        stored_value = tl.load(pair_values_ptr + pair_slot)
        pair_candidate = tl.where(
            (pair_candidate < 0) & (stored_key == pair_key),
            stored_value,
            pair_candidate,
        )
    fallback_candidate = tl.where(
        pair_candidate >= 0,
        pair_candidate,
        transition_candidate,
    )
    output = tl.where(
        best_length > 0,
        ngram_candidate,
        fallback_candidate,
    )
    tl.store(output_ptr + row, output)


class DraftHybridSecondTokenState:
    """Build a strict-verifier-safe second proposal from verified history."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        history_width: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        if history_width < 16:
            raise ValueError("hybrid Draft history width must be at least 16")
        self.max_batch_size = max_batch_size
        self.history_width = history_width
        self.vocab_size = vocab_size
        pin_memory = is_pin_memory_available()

        self.history_cpu = torch.zeros(
            (max_batch_size, history_width),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.lengths_cpu = torch.zeros(
            max_batch_size,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.transition_cpu = torch.full(
            (vocab_size,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.transition_numpy = self.transition_cpu.numpy()
        self.transition_counts: dict[int, Counter[int]] = {}

        self.pair_table_size = 1 << 18
        self.pair_table_mask = self.pair_table_size - 1
        self.pair_keys_cpu = torch.full(
            (self.pair_table_size,),
            -1,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.pair_values_cpu = torch.full(
            (self.pair_table_size,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.pair_keys_numpy = self.pair_keys_cpu.numpy()
        self.pair_values_numpy = self.pair_values_cpu.numpy()
        self.pair_counts: dict[int, Counter[int]] = {}

        self.history = torch.zeros(
            (max_batch_size, history_width),
            dtype=torch.int32,
            device=device,
        )
        self.lengths = torch.zeros(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self.transition = torch.full(
            (vocab_size,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.pair_keys = torch.full(
            (self.pair_table_size,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.pair_values = torch.full(
            (self.pair_table_size,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.discard = torch.zeros(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self.second_tokens = torch.empty(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self._processed_lengths: dict[str, int] = {}

    def _learn_transition(self, source: int, target: int) -> None:
        if not (0 <= source < self.vocab_size and 0 <= target < self.vocab_size):
            return
        counts = self.transition_counts.setdefault(source, Counter())
        counts[target] += 1
        current = int(self.transition_numpy[source])
        if current < 0 or counts[target] > counts[current]:
            self.transition_numpy[source] = target

    def _learn_pair(self, base: int, first: int, target: int) -> None:
        if not (
            0 <= base < self.vocab_size
            and 0 <= first < self.vocab_size
            and 0 <= target < self.vocab_size
        ):
            return
        key = base * self.vocab_size + first
        slot = (base * 1315423911 + first * 2654435761) & self.pair_table_mask
        for probe in range(4):
            candidate_slot = (slot + probe) & self.pair_table_mask
            stored_key = int(self.pair_keys_numpy[candidate_slot])
            if stored_key not in {-1, key}:
                continue
            if stored_key == -1:
                self.pair_keys_numpy[candidate_slot] = key
            counts = self.pair_counts.setdefault(key, Counter())
            counts[target] += 1
            current = int(self.pair_values_numpy[candidate_slot])
            if current < 0 or counts[target] > counts[current]:
                self.pair_values_numpy[candidate_slot] = target
            return

    def prepare(self, input_batch: Any, num_rows: int) -> None:
        """Refresh graph inputs and learn transitions from verified tokens."""
        if num_rows > self.max_batch_size:
            raise RuntimeError(
                "hybrid Draft batch exceeds its persistent buffer: "
                f"{num_rows} > {self.max_batch_size}"
            )

        active_req_ids = input_batch.req_ids[:num_rows]
        token_ids = input_batch.token_ids_cpu
        lengths = input_batch.num_tokens_no_spec
        max_context = self.history_width - 2
        self.history_cpu[:num_rows].zero_()

        for row, req_id in enumerate(active_req_ids):
            full_length = int(lengths[row])
            previous_length = self._processed_lengths.get(req_id, 0)
            if previous_length > full_length:
                previous_length = 0
            start = max(previous_length, 1)
            for position in range(start, full_length):
                source = int(token_ids[row, position - 1])
                target = int(token_ids[row, position])
                self._learn_transition(source, target)
                if position >= 2:
                    self._learn_pair(
                        int(token_ids[row, position - 2]),
                        source,
                        target,
                    )
            self._processed_lengths[req_id] = full_length

            retained = min(full_length, max_context)
            if retained:
                source_start = full_length - retained
                self.history_cpu[row, :retained].copy_(
                    input_batch.token_ids_cpu_tensor[
                        row,
                        source_start:full_length,
                    ]
                )
            self.lengths_cpu[row] = retained

        self.history[:num_rows].copy_(
            self.history_cpu[:num_rows],
            non_blocking=True,
        )
        self.lengths[:num_rows].copy_(
            self.lengths_cpu[:num_rows],
            non_blocking=True,
        )
        self.transition.copy_(self.transition_cpu, non_blocking=True)
        self.pair_keys.copy_(self.pair_keys_cpu, non_blocking=True)
        self.pair_values.copy_(self.pair_values_cpu, non_blocking=True)

    def propose(
        self,
        base_token_ids: torch.Tensor,
        first_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return one target-vocabulary candidate without host synchronization."""
        num_rows = first_token_ids.shape[0]
        block = triton.next_power_of_2(self.history_width)
        _hybrid_second_token_kernel[(num_rows,)](
            self.history,
            self.lengths,
            base_token_ids,
            first_token_ids,
            self.transition,
            self.pair_keys,
            self.pair_values,
            self.second_tokens,
            self.history.stride(0),
            self.vocab_size,
            self.pair_table_mask,
            history_width=self.history_width,
            BLOCK=block,
            PAIR_PROBES=4,
        )
        return self.second_tokens[:num_rows]
