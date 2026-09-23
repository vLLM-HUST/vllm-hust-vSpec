"""Compact EAGLE Draft LM-head projection with full token-ID mapping."""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from .eagle_target import load_active_vocab_ids

PATCH_MARKER = "_vllm_hust_vspec_draft_vocab_patched"
QUANT_CHUNK_SIZE = 64000


class ActiveVocabLogits:
    """Minimal logits facade used by the native serial proposer."""

    def __init__(self, logits: torch.Tensor, active_ids: torch.Tensor) -> None:
        self.logits = logits
        self.active_ids = active_ids

    @property
    def shape(self) -> torch.Size:
        return self.logits.shape

    def argmax(self, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        compact_ids = self.logits.argmax(dim=dim, keepdim=keepdim)
        return self.active_ids[compact_ids]

    def contiguous(self) -> ActiveVocabLogits:
        self.logits = self.logits.contiguous()
        return self

    def __getitem__(self, key: Any) -> ActiveVocabLogits:
        return ActiveVocabLogits(self.logits[key], self.active_ids)


class ChunkedQuantizedLogits:
    """Lazy quantized projection that reduces each vocabulary chunk in place."""

    def __init__(
        self,
        hidden_states: torch.Tensor,
        chunks: list[tuple[int, torch.Tensor, torch.Tensor]],
        bias: torch.Tensor | None,
        use_w8a8: bool,
    ) -> None:
        self.hidden_states = hidden_states
        self.chunks = chunks
        self.bias = bias
        self.use_w8a8 = use_w8a8
        self.vocab_size = sum(chunk[1].shape[1] for chunk in chunks)

    @property
    def shape(self) -> torch.Size:
        return torch.Size((*self.hidden_states.shape[:-1], self.vocab_size))

    def argmax(self, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        if dim not in (-1, len(self.shape) - 1):
            raise ValueError("Chunked quantized logits only support vocabulary argmax")

        quant_hidden_states, pertoken_scale = _quantize_hidden_states(
            self.hidden_states,
            self.use_w8a8,
        )
        best_values = None
        best_ids = None
        for start, quant_weight, quant_scale in self.chunks:
            output = _compute_quantized_chunk(
                self.hidden_states,
                quant_hidden_states,
                pertoken_scale,
                start,
                quant_weight,
                quant_scale,
                self.bias,
                self.use_w8a8,
            )
            chunk_values, chunk_ids = output.max(dim=-1)
            chunk_ids = chunk_ids.add(start)
            if best_values is None:
                best_values = chunk_values
                best_ids = chunk_ids
                continue
            replace = chunk_values > best_values
            best_values = torch.where(replace, chunk_values, best_values)
            best_ids = torch.where(replace, chunk_ids, best_ids)

        if best_ids is None:
            raise RuntimeError("Quantized LM head has no vocabulary chunks")
        return best_ids.unsqueeze(-1) if keepdim else best_ids

    def contiguous(self) -> ChunkedQuantizedLogits:
        self.hidden_states = self.hidden_states.contiguous()
        return self

    def __getitem__(self, key: Any) -> ChunkedQuantizedLogits:
        return ChunkedQuantizedLogits(
            self.hidden_states[key],
            self.chunks,
            self.bias,
            self.use_w8a8,
        )


def quantize_active_lm_head_w8a16(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_size, input_size = weight.shape
    quant_weight = torch.empty(
        (input_size, output_size),
        dtype=torch.int8,
        device=weight.device,
    )
    quant_scale = torch.empty(
        output_size,
        dtype=weight.dtype,
        device=weight.device,
    )
    for start in range(0, output_size, 4096):
        end = min(start + 4096, output_size)
        chunk = weight[start:end].float()
        scale = chunk.abs().amax(dim=1).clamp_min_(1e-8).div_(127.0)
        quant_scale[start:end].copy_(scale.to(weight.dtype))
        quant_weight[:, start:end].copy_(
            chunk.div_(scale.unsqueeze(1)).round_().clamp_(-127, 127).to(torch.int8).transpose(0, 1)
        )
    return quant_weight, quant_scale


def quantize_active_lm_head_chunks(
    weight: torch.Tensor,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    chunks = []
    for start in range(0, weight.shape[0], QUANT_CHUNK_SIZE):
        quant_weight, quant_scale = quantize_active_lm_head_w8a16(
            weight[start : start + QUANT_CHUNK_SIZE]
        )
        chunks.append((start, quant_weight, quant_scale))
    return chunks


def _quantize_hidden_states(
    hidden_states: torch.Tensor,
    use_w8a8: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not use_w8a8:
        return None, None
    import torch_npu

    return torch_npu.npu_dynamic_quant(
        hidden_states,
        dst_type=torch.int8,
    )


def _compute_quantized_chunk(
    hidden_states: torch.Tensor,
    quant_hidden_states: torch.Tensor | None,
    pertoken_scale: torch.Tensor | None,
    start: int,
    quant_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    bias: torch.Tensor | None,
    use_w8a8: bool,
) -> torch.Tensor:
    import torch_npu

    end = start + quant_weight.shape[1]
    chunk_bias = bias[start:end] if bias is not None else None
    if use_w8a8:
        assert quant_hidden_states is not None
        assert pertoken_scale is not None
        return torch_npu.npu_quant_matmul(
            quant_hidden_states,
            quant_weight,
            quant_scale,
            pertoken_scale=pertoken_scale,
            bias=chunk_bias,
            output_dtype=hidden_states.dtype,
        )
    return torch_npu.npu_weight_quant_batchmatmul(
        hidden_states,
        quant_weight,
        quant_scale,
        bias=chunk_bias,
    )


def _compute_quantized_logits(
    hidden_states: torch.Tensor,
    chunks: list[tuple[int, torch.Tensor, torch.Tensor]],
    bias: torch.Tensor | None,
    use_w8a8: bool,
) -> torch.Tensor:
    quant_hidden_states, pertoken_scale = _quantize_hidden_states(
        hidden_states,
        use_w8a8,
    )
    outputs = []
    for start, quant_weight, quant_scale in chunks:
        outputs.append(
            _compute_quantized_chunk(
                hidden_states,
                quant_hidden_states,
                pertoken_scale,
                start,
                quant_weight,
                quant_scale,
                bias,
                use_w8a8,
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


def _configure_draft_active_vocab(proposer: Any) -> None:
    active_ids_path = os.environ.get("VLLM_ASCEND_EAGLE_DRAFT_ACTIVE_VOCAB_IDS_PATH")
    active_size = int(os.environ.get("VLLM_ASCEND_EAGLE_DRAFT_ACTIVE_VOCAB_SIZE", "0"))
    use_w8a16 = os.environ.get("VLLM_ASCEND_EAGLE_DRAFT_LM_HEAD_W8A16") == "1"
    use_w8a8 = os.environ.get("VLLM_ASCEND_EAGLE_DRAFT_LM_HEAD_W8A8") == "1"
    if use_w8a16 and use_w8a8:
        raise RuntimeError("EAGLE Draft LM head cannot enable W8A16 and W8A8 together")
    if use_w8a16 or use_w8a8:
        native_chunks = getattr(
            proposer,
            "_eagle_draft_active_lm_head_w8a16_chunks",
            None,
        )
        native_weight = getattr(
            proposer,
            "_eagle_draft_active_lm_head_w8a16_weight",
            None,
        )
        if native_chunks or native_weight is not None:
            return
    proposer._eagle_draft_active_vocab_ids = None
    proposer._eagle_draft_active_lm_head_weight = None
    proposer._eagle_draft_active_lm_head_bias = None
    proposer._eagle_draft_active_lm_head_w8a16_chunks = None
    if not active_ids_path and active_size == 0 and not use_w8a16 and not use_w8a8:
        return
    if proposer.method != "eagle":
        raise RuntimeError("EAGLE Draft active vocabulary requires method=eagle")
    if proposer.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("EAGLE Draft active vocabulary currently requires TP=1")
    if proposer.vllm_config.quant_config is not None:
        raise RuntimeError("EAGLE Draft active vocabulary does not support model quantization")
    lm_head = proposer.model.lm_head
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("EAGLE Draft active vocabulary requires a 2-D LM head")
    vocab_size = int(weight.shape[0])
    if active_ids_path:
        active_ids = load_active_vocab_ids(
            Path(active_ids_path),
            vocab_size,
            weight.device,
        )
    elif active_size > 0:
        if active_size < 1024 or active_size >= vocab_size:
            raise RuntimeError(
                f"EAGLE Draft active vocabulary size must be in [1024, {vocab_size})"
            )
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
        active_ids = torch.arange(
            vocab_size,
            dtype=torch.long,
            device=weight.device,
        )
    if active_ids.numel() < 1024:
        raise RuntimeError("EAGLE Draft active vocabulary requires at least 1024 IDs")

    active_weight = weight.index_select(0, active_ids).contiguous()
    bias = getattr(lm_head, "bias", None)
    active_bias = (
        bias.index_select(0, active_ids).contiguous() if isinstance(bias, torch.Tensor) else None
    )
    proposer._eagle_draft_active_vocab_ids = active_ids
    proposer._eagle_draft_active_lm_head_bias = active_bias
    quant_chunks = None
    projection_weight = active_weight
    if use_w8a16 or use_w8a8:
        if active_ids.numel() % 64:
            mode = "W8A8" if use_w8a8 else "W8A16"
            raise RuntimeError(
                f"EAGLE Draft {mode} LM head requires an active vocabulary size divisible by 64"
            )
        quant_chunks = quantize_active_lm_head_chunks(active_weight)
        proposer._eagle_draft_active_lm_head_w8a16_chunks = quant_chunks
        proposer._eagle_draft_active_lm_head_weight = None
        projection_weight = None
    else:
        proposer._eagle_draft_active_lm_head_weight = active_weight

    model = proposer.model

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
        logits_processor._gather_logits = MethodType(
            gather_logits,
            logits_processor,
        )


def apply_draft_active_vocab_patch() -> bool:
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )

    if getattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, False):
        return False
    original_load_model = AscendSpecDecodeBaseProposer.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        _configure_draft_active_vocab(self)
        return result

    AscendSpecDecodeBaseProposer.load_model = load_model
    setattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, True)
    return True
