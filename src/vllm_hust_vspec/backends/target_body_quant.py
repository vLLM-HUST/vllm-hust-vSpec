"""Online weight-only quantization for the target transformer body."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from .eagle_body_quant import (
    _install_compiler_compatibility,
    _quantize_weight,
    _WeightOnlyLinearMethod,
)

logger = logging.getLogger(__name__)

PATCH_MARKER = "_vllm_hust_vspec_target_body_w8a16_patched"
CONFIGURED_MARKER = "_vllm_hust_vspec_target_body_w8a16_configured"


class _DynamicInt8LinearMethod:
    def apply(
        self,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch_npu

        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(
            x,
            dst_type=torch.int8,
        )
        return torch_npu.npu_quant_matmul(
            quant_x,
            layer._vspec_w8a16_weight,
            layer._vspec_w8a16_scale,
            pertoken_scale=pertoken_scale,
            bias=(layer._vspec_w8a16_bias if bias is not None else None),
            output_dtype=x.dtype,
        )


def _unwrap_model(model: Any) -> Any:
    unwrap = getattr(model, "unwrap", None)
    return unwrap() if callable(unwrap) else model


def _configure_target_body_quantization(runner: Any, mode: str) -> None:
    model = _unwrap_model(runner.model)
    if getattr(model, CONFIGURED_MARKER, False):
        return
    setattr(model, CONFIGURED_MARKER, True)
    if runner.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("Target body W8A16 currently requires TP=1")
    if runner.vllm_config.quant_config is not None:
        raise RuntimeError("Target body W8A16 does not support an already quantized model")

    from vllm.model_executor.layers.linear import LinearBase

    scope = os.environ.get("VSPEC_TARGET_BODY_QUANT_SCOPE", "all")
    if scope not in {"all", "mlp", "attention"}:
        raise RuntimeError("VSPEC_TARGET_BODY_QUANT_SCOPE must be all, mlp, or attention")
    configured: list[str] = []
    released_bytes = 0
    for name, layer in model.named_modules():
        if not isinstance(layer, LinearBase) or name.endswith("lm_head"):
            continue
        if scope == "mlp" and ".mlp." not in name:
            continue
        if scope == "attention" and ".self_attn." not in name:
            continue
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if weight.shape[0] % 64 or weight.shape[1] % 64:
            raise RuntimeError(
                "Target body W8A16 requires dimensions divisible by 64: "
                f"{name}={tuple(weight.shape)}"
            )
        quant_weight, quant_scale = _quantize_weight(weight)
        layer.register_buffer("_vspec_w8a16_weight", quant_weight)
        layer.register_buffer("_vspec_w8a16_scale", quant_scale)
        bias = getattr(layer, "bias", None)
        quant_bias = bias.float().contiguous() if isinstance(bias, torch.Tensor) else None
        layer.register_buffer("_vspec_w8a16_bias", quant_bias)
        released_bytes += weight.numel() * weight.element_size()
        layer.register_parameter("weight", None)
        layer.quant_method = (
            _DynamicInt8LinearMethod() if mode == "w8a8" else _WeightOnlyLinearMethod()
        )
        configured.append(name)

    if not configured:
        raise RuntimeError("Target body W8A16 found no eligible linear layers")
    logger.info(
        "Enabled online Target body %s scope=%s for %d layers; released %.2f GiB",
        mode.upper(),
        scope,
        len(configured),
        released_bytes / float(2**30),
    )


def apply_target_body_quantization_patch(mode: str = "w8a16") -> bool:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if mode not in {"w8a16", "w8a8"}:
        raise ValueError(f"unsupported Target body quantization mode: {mode}")
    if getattr(NPUModelRunner, PATCH_MARKER, False):
        return False
    _install_compiler_compatibility()
    original_load_model = NPUModelRunner.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        _configure_target_body_quantization(self, mode)
        return result

    NPUModelRunner.load_model = load_model
    setattr(NPUModelRunner, PATCH_MARKER, True)
    return True
