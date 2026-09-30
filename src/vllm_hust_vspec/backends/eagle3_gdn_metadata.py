"""Share group-invariant Qwen3.5 GDN metadata during EAGLE3 decode."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

import torch

PATCH_MARKER = "_vllm_hust_vspec_eagle3_gdn_metadata_patched"


@dataclass
class _SharedMetadataState:
    leader: Any | None = None
    generation: int = 0
    metadata: Any | None = None
    gather_indices: torch.Tensor | None = None
    logged: bool = False


def _is_uniform_spec_decode(metadata: Any, num_reqs: int) -> bool:
    return bool(
        metadata is not None
        and metadata.num_prefills == 0
        and metadata.num_decodes == 0
        and metadata.num_spec_decodes == num_reqs
        and metadata.spec_sequence_masks is not None
        and metadata.spec_query_start_loc is not None
        and metadata.spec_state_indices_tensor is not None
        and metadata.num_accepted_tokens is not None
    )


def apply_eagle3_gdn_metadata_sharing_patch() -> bool:
    from vllm.logger import logger
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder,
        GDNSpecCausalConv1dMetadata,
        GDNSpecDecodeMetadata,
    )

    builder_cls = AscendGDNAttentionMetadataBuilder
    if getattr(builder_cls, PATCH_MARKER, False):
        return False

    original_init = builder_cls.__init__
    original_build = builder_cls.build
    states: dict[int, _SharedMetadataState] = {}

    @wraps(original_init)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        state = states.setdefault(id(self.vllm_config), _SharedMetadataState())
        if state.leader is None:
            state.leader = self
        self._vspec_gdn_shared_state = state
        self._vspec_gdn_seen_generation = 0

    @wraps(original_build)
    def build(
        self: Any,
        common_prefix_len: int,
        common_attn_metadata: Any,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> Any:
        state: _SharedMetadataState = self._vspec_gdn_shared_state
        if state.leader is self:
            metadata = original_build(
                self,
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                fast_build,
            )
            state.generation += 1
            state.metadata = metadata
            state.gather_indices = None
            self._vspec_gdn_seen_generation = state.generation
            if _is_uniform_spec_decode(metadata, common_attn_metadata.num_reqs):
                kv_spec = self.kv_cache_spec
                start_indices = (
                    (common_attn_metadata.seq_lens - 1) // kv_spec.block_size
                ).clamp(min=0)
                offsets = torch.arange(
                    1 + kv_spec.num_speculative_blocks,
                    device=common_attn_metadata.block_table_tensor.device,
                    dtype=torch.int32,
                )
                state.gather_indices = (
                    start_indices.unsqueeze(1) + offsets
                ).to(torch.int64)
            return metadata

        source = state.metadata
        can_share = bool(
            self._vspec_gdn_seen_generation != state.generation
            and state.gather_indices is not None
            and _is_uniform_spec_decode(source, common_attn_metadata.num_reqs)
            and num_accepted_tokens is not None
        )
        if not can_share:
            return original_build(
                self,
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                fast_build,
            )

        self._vspec_gdn_seen_generation = state.generation
        num_reqs = source.num_spec_decodes
        graph_batch_size = source.spec_state_indices_tensor.size(0)
        gathered_block_table = torch.gather(
            common_attn_metadata.block_table_tensor,
            1,
            state.gather_indices,
        )

        state_indices = self.spec_state_indices_tensor[:graph_batch_size]
        state_indices[:num_reqs].copy_(
            gathered_block_table[:num_reqs, : self.num_spec + 1],
            non_blocking=True,
        )
        state_indices[num_reqs:].fill_(0)

        accepted = self.num_accepted_tokens[:graph_batch_size]
        accepted[:num_reqs].copy_(
            num_accepted_tokens[:num_reqs],
            non_blocking=True,
        )
        accepted[num_reqs:].fill_(1)

        sequence_masks = self.spec_sequence_masks[:graph_batch_size]
        sequence_masks[:num_reqs].fill_(True)
        sequence_masks[num_reqs:].fill_(False)

        query_start_loc = self.spec_query_start_loc[: graph_batch_size + 1]
        query_start_loc.copy_(source.spec_query_start_loc, non_blocking=True)
        actual_seq_lengths = self.spec_actual_seq_lengths[: graph_batch_size + 1]
        actual_seq_lengths.copy_(
            source.spec_decode_metadata.actual_seq_lengths,
            non_blocking=True,
        )

        spec_token_indx = self.spec_token_indx[: source.spec_token_indx.size(0)]
        spec_token_indx.copy_(source.spec_token_indx, non_blocking=True)
        non_spec_token_indx = self.non_spec_token_indx[:0]

        metadata = replace(
            source,
            spec_query_start_loc=query_start_loc,
            spec_state_indices_tensor=state_indices,
            spec_sequence_masks=sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=accepted,
        )
        metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
            spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
                query_start_loc=query_start_loc,
                cache_indices=state_indices,
                num_accepted_tokens=accepted,
            ),
            actual_seq_lengths=actual_seq_lengths,
        )
        metadata.non_spec_prefill_metadata = None
        metadata.non_spec_decode_metadata = None
        if not state.logged:
            logger.info(
                "vSpec EAGLE3 enabled shared GDN speculative metadata for "
                "uniform Graph batches"
            )
            state.logged = True
        return metadata

    builder_cls.__init__ = init
    builder_cls.build = build
    setattr(builder_cls, PATCH_MARKER, True)
    return True
