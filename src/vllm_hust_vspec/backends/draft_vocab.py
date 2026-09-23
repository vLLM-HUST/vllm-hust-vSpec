"""Compact candidate projection for the serial Draft backend."""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from .eagle_draft import (
    ActiveVocabLogits,
    ChunkedQuantizedLogits,
    _compute_quantized_logits,
    quantize_active_lm_head_chunks,
)
from .eagle_target import load_active_vocab_ids

DRAFT_ACTIVE_VOCAB_SIZE = "VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_SIZE"
DRAFT_ACTIVE_VOCAB_IDS_PATH = "VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_IDS_PATH"
DRAFT_LM_HEAD_W8A16 = "VLLM_ASCEND_DRAFT_LM_HEAD_W8A16"
DRAFT_LM_HEAD_W8A8 = "VLLM_ASCEND_DRAFT_LM_HEAD_W8A8"


def _configure_serial_draft_active_vocab(proposer: Any) -> bool:
    """Install a strict-verification-safe compact candidate LM head."""
    if getattr(proposer, "_vspec_draft_vocab_configured", False):
        return False

    active_ids_path = os.environ.get(DRAFT_ACTIVE_VOCAB_IDS_PATH)
    raw_active_size = os.environ.get(DRAFT_ACTIVE_VOCAB_SIZE, "0")
    try:
        active_size = int(raw_active_size)
    except ValueError as exc:
        raise RuntimeError(f"{DRAFT_ACTIVE_VOCAB_SIZE} must be an integer") from exc
    use_w8a16 = os.environ.get(DRAFT_LM_HEAD_W8A16) == "1"
    use_w8a8 = os.environ.get(DRAFT_LM_HEAD_W8A8) == "1"
    requested = bool(active_ids_path or active_size or use_w8a16 or use_w8a8)
    if not requested:
        proposer._vspec_draft_vocab_configured = True
        return False
    if active_size < 0:
        raise RuntimeError(f"{DRAFT_ACTIVE_VOCAB_SIZE} must be non-negative")
    if active_size and active_ids_path:
        raise RuntimeError("Draft active vocabulary size and ID path are mutually exclusive")
    if use_w8a16 and use_w8a8:
        raise RuntimeError("Draft LM head cannot enable W8A16 and W8A8 together")
    if proposer.method != "draft_model":
        raise RuntimeError("Draft active vocabulary requires method=draft_model")
    if proposer.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("Draft active vocabulary currently requires TP=1")
    if proposer.vllm_config.quant_config is not None:
        raise RuntimeError("Draft active vocabulary does not support whole-model quantization")

    lm_head = proposer.model.lm_head
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("Draft active vocabulary requires a 2-D LM head")
    vocab_size = int(weight.shape[0])
    proposer._vspec_draft_full_vocab_size = vocab_size
    if active_ids_path:
        active_ids = load_active_vocab_ids(
            Path(active_ids_path),
            vocab_size,
            weight.device,
        )
    elif active_size:
        if active_size < 1024 or active_size >= vocab_size:
            raise RuntimeError(f"Draft active vocabulary size must be in [1024, {vocab_size})")
        special_tail_size = min(512, active_size // 4)
        prefix_size = active_size - special_tail_size
        active_ids = torch.cat(
            (
                torch.arange(prefix_size, device=weight.device),
                torch.arange(
                    vocab_size - special_tail_size,
                    vocab_size,
                    device=weight.device,
                ),
            )
        )
    else:
        active_ids = torch.arange(vocab_size, device=weight.device)

    if active_ids.numel() < 1024:
        raise RuntimeError("Draft active vocabulary requires at least 1024 IDs")
    if (use_w8a16 or use_w8a8) and active_ids.numel() % 64:
        mode = "W8A8" if use_w8a8 else "W8A16"
        raise RuntimeError(
            f"Draft {mode} LM head requires an active vocabulary size divisible by 64"
        )

    active_weight = weight.index_select(0, active_ids).contiguous()
    bias = getattr(lm_head, "bias", None)
    active_bias = (
        bias.index_select(0, active_ids).contiguous() if isinstance(bias, torch.Tensor) else None
    )
    proposer._eagle_draft_active_vocab_ids = active_ids
    proposer._eagle_draft_active_lm_head_bias = active_bias
    proposer._eagle_draft_active_lm_head_weight = active_weight
    proposer._eagle_draft_active_lm_head_w8a16_weight = None
    proposer._eagle_draft_active_lm_head_w8a16_scale = None
    proposer._eagle_draft_active_lm_head_w8a16_chunks = None
    proposer.eagle_draft_lm_head_w8a8 = use_w8a8
    if use_w8a16 or use_w8a8:
        proposer._eagle_draft_active_lm_head_w8a16_chunks = quantize_active_lm_head_chunks(
            active_weight
        )
        proposer._eagle_draft_active_lm_head_weight = None

    model = proposer.model
    quant_chunks = proposer._eagle_draft_active_lm_head_w8a16_chunks
    projection_weight = proposer._eagle_draft_active_lm_head_weight

    def compute_active_logits(
        _model: Any,
        hidden_states: torch.Tensor,
    ) -> ActiveVocabLogits:
        if quant_chunks is None:
            assert projection_weight is not None
            compact_logits = torch.nn.functional.linear(
                hidden_states,
                projection_weight,
                active_bias,
            )
        elif os.environ.get("VSPEC_DRAFT_CHUNKED_QUANT_ARGMAX") == "1":
            compact_logits = ChunkedQuantizedLogits(
                hidden_states,
                quant_chunks,
                active_bias,
                use_w8a8,
            )
        else:
            compact_logits = _compute_quantized_logits(
                hidden_states,
                quant_chunks,
                active_bias,
                use_w8a8,
            )
        return ActiveVocabLogits(compact_logits, active_ids)

    if not hasattr(model, "_vspec_original_compute_logits"):
        model._vspec_original_compute_logits = model.compute_logits
    model.compute_logits = MethodType(compute_active_logits, model)

    logits_processor = getattr(getattr(model, "model", None), "logits_processor", None)
    if logits_processor is not None and not hasattr(
        logits_processor,
        "_vspec_original_gather_logits",
    ):
        original_gather = logits_processor._gather_logits

        def gather_logits(_processor: Any, logits: Any) -> Any:
            if isinstance(logits, ActiveVocabLogits):
                return logits
            return original_gather(logits)

        logits_processor._vspec_original_gather_logits = original_gather
        logits_processor._gather_logits = MethodType(gather_logits, logits_processor)

    proposer._vspec_draft_vocab_configured = True
    return True
