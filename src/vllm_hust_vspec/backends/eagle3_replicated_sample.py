"""Avoid redundant TP collectives for replicated EAGLE3 draft heads."""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch

PATCH_MARKER = "_vllm_hust_vspec_eagle3_replicated_sample_patched"


def _sample_replicated_draft_logits(
    logits: torch.Tensor,
    draft_id_to_target_id: torch.Tensor,
) -> torch.Tensor:
    """Map a local draft-vocabulary argmax to the target vocabulary."""
    draft_token = logits.argmax(dim=-1)
    bias = torch.index_select(
        draft_id_to_target_id,
        dim=0,
        index=draft_token.reshape(-1),
    ).view_as(draft_token)
    return draft_token + bias


def apply_eagle3_replicated_sample_patch() -> bool:
    """Use local argmax when every target TP rank owns a full draft model.

    The Ascend EAGLE3 loader creates one draft-TP group per target rank when
    ``draft_tensor_parallel_size=1``. The draft body and its 32K LM head are
    therefore replicated, while the generic sampler still consults the target
    TP group and performs two all-gathers for every proposed token. Replicated
    hidden states and weights produce identical logits, so those collectives
    cannot affect the selected token.
    """
    from vllm.logger import logger
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )

    original = AscendSpecDecodeBaseProposer.compute_draft_token_ids
    if getattr(original, PATCH_MARKER, False):
        return False

    @wraps(original)
    def compute_draft_token_ids(
        self: Any,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mapping = getattr(self.model, "draft_id_to_target_id", None)
        draft_tp_size = self.speculative_config.draft_tensor_parallel_size
        if self.method != "eagle3" or draft_tp_size != 1 or mapping is None:
            return original(self, hidden_states, sampling_metadata)

        logits = self.model.logits_processor(self.model.lm_head, hidden_states)
        return _sample_replicated_draft_logits(logits.contiguous(), mapping), None

    setattr(compute_draft_token_ids, PATCH_MARKER, True)
    AscendSpecDecodeBaseProposer.compute_draft_token_ids = compute_draft_token_ids
    logger.info(
        "vSpec enabled collective-free EAGLE3 sampling for replicated draft TP=1"
    )
    return True
