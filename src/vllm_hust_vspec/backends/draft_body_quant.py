"""Online weight-only quantization for a serial Draft transformer body."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .eagle_body_quant import (
    _install_compiler_compatibility,
    _quantize_weight,
    _WeightOnlyLinearMethod,
)

logger = logging.getLogger(__name__)

PATCH_MARKER = "_vllm_hust_vspec_draft_body_w8a16_patched"
CONFIGURED_MARKER = "_vllm_hust_vspec_draft_body_w8a16_configured"


def _configure_draft_body_quantization(proposer: Any) -> None:
    if getattr(proposer, CONFIGURED_MARKER, False):
        return
    setattr(proposer, CONFIGURED_MARKER, True)
    if proposer.method != "draft_model":
        raise RuntimeError("Draft body W8A16 requires method=draft_model")
    if proposer.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("Draft body W8A16 currently requires target TP=1")
    if proposer.speculative_config.draft_parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("Draft body W8A16 currently requires draft TP=1")
    if proposer.vllm_config.quant_config is not None:
        raise RuntimeError("Draft body W8A16 does not support a quantized target model")

    from vllm.model_executor.layers.linear import LinearBase

    draft_body = getattr(proposer.model, "model", None)
    if draft_body is None:
        raise RuntimeError("Draft body W8A16 cannot locate the transformer body")

    configured: list[str] = []
    released_bytes = 0
    for name, layer in draft_body.named_modules():
        if not isinstance(layer, LinearBase):
            continue
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if weight.shape[0] % 64 or weight.shape[1] % 64:
            raise RuntimeError(
                "Draft body W8A16 requires dimensions divisible by 64: "
                f"{name}={tuple(weight.shape)}"
            )
        quant_weight, quant_scale = _quantize_weight(weight)
        layer.register_buffer("_vspec_w8a16_weight", quant_weight)
        layer.register_buffer("_vspec_w8a16_scale", quant_scale)
        bias = getattr(layer, "bias", None)
        quant_bias = (
            bias.to(dtype=weight.dtype).contiguous() if isinstance(bias, torch.Tensor) else None
        )
        layer.register_buffer("_vspec_w8a16_bias", quant_bias)
        released_bytes += weight.numel() * weight.element_size()
        layer.register_parameter("weight", None)
        layer.quant_method = _WeightOnlyLinearMethod()
        configured.append(name)

    if not configured:
        raise RuntimeError("Draft body W8A16 found no eligible linear layers")
    logger.info(
        "Enabled online Draft body W8A16 for %d layers; released %.2f GiB",
        len(configured),
        released_bytes / float(2**30),
    )


def apply_draft_body_quantization_patch() -> bool:
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )

    if getattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, False):
        return False
    _install_compiler_compatibility()
    original_load_model = AscendSpecDecodeBaseProposer.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        _configure_draft_body_quantization(self)
        return result

    AscendSpecDecodeBaseProposer.load_model = load_model
    setattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, True)
    return True
