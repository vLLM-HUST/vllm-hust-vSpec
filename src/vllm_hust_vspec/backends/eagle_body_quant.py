"""Online weight-only quantization for the Qwen2 EAGLE draft body."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

PATCH_MARKER = "_vllm_hust_vspec_eagle_body_w8a16_patched"
CONFIGURED_MARKER = "_vllm_hust_vspec_eagle_body_w8a16_configured"
COMPILER_COMPAT_MARKER = "_vllm_hust_vspec_weight_quant_compat_patched"


class _WeightOnlyLinearMethod:
    def apply(
        self,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch_npu

        quant_bias = layer._vspec_w8a16_bias if bias is not None else None
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            layer._vspec_w8a16_weight,
            layer._vspec_w8a16_scale,
            bias=quant_bias,
        )


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _selected_layers() -> set[str] | None:
    raw = os.environ.get("VSPEC_EAGLE_DRAFT_BODY_W8A16_LAYERS", "")
    selected = {name.strip() for name in raw.split(",") if name.strip()}
    return selected or None


def _install_compiler_compatibility() -> None:
    """Bridge a torch_npu config mismatch in weight-quant graph lowering."""
    from npugraph_ex.configs._option_base import OptionValue
    from npugraph_ex.configs.experimental_config import _ExperimentalConfig

    if getattr(_ExperimentalConfig, COMPILER_COMPAT_MARKER, False):
        return
    original_init = _ExperimentalConfig.__init__
    original_setattr = _ExperimentalConfig.__setattr__

    def init(self: Any) -> None:
        original_init(self)
        # torch_npu's weight-quant lowering still writes this retired option.
        # Keep it local-only: the current config serializer intentionally does
        # not forward it to the newer npugraph_ex backend.
        object.__setattr__(
            self,
            "enable_view_optimize",
            OptionValue(True, [True, False]),
        )
        self._fixed_attrs.append("enable_view_optimize")

    def set_option(self: Any, key: str, value: Any) -> None:
        if key == "enable_view_optimize":
            current = getattr(self, key, None)
            if isinstance(current, OptionValue):
                current.value = value
            else:
                # The compiler config is normally instantiated before plugins
                # are loaded. Keep the retired field local on that instance.
                object.__setattr__(self, key, value)
            return
        original_setattr(self, key, value)

    _ExperimentalConfig.__init__ = init
    _ExperimentalConfig.__setattr__ = set_option
    setattr(_ExperimentalConfig, COMPILER_COMPAT_MARKER, True)


def _configure_body_quantization(proposer: Any) -> None:
    if getattr(proposer, CONFIGURED_MARKER, False):
        return
    setattr(proposer, CONFIGURED_MARKER, True)
    if proposer.method != "eagle":
        raise RuntimeError("EAGLE body W8A16 requires method=eagle")
    if proposer.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("EAGLE body W8A16 currently requires TP=1")
    if proposer.vllm_config.quant_config is not None:
        raise RuntimeError("EAGLE body W8A16 does not support a quantized target model")

    from vllm.model_executor.layers.linear import LinearBase

    draft_body = getattr(proposer.model, "model", None)
    if draft_body is None:
        raise RuntimeError("EAGLE body W8A16 cannot locate the draft body")
    selected = _selected_layers()
    configured: list[str] = []
    available: list[str] = []
    for name, layer in draft_body.named_modules():
        if not isinstance(layer, LinearBase):
            continue
        available.append(name)
        if selected is not None and name not in selected:
            continue
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError(f"EAGLE body W8A16 requires a 2-D weight for {name}")
        if weight.shape[0] % 64 or weight.shape[1] % 64:
            raise RuntimeError(
                f"EAGLE body W8A16 requires dimensions divisible by 64: "
                f"{name}={tuple(weight.shape)}"
            )
        quant_weight, quant_scale = _quantize_weight(weight)
        layer.register_buffer("_vspec_w8a16_weight", quant_weight)
        layer.register_buffer("_vspec_w8a16_scale", quant_scale)
        bias = getattr(layer, "bias", None)
        quant_bias = bias.float().contiguous() if isinstance(bias, torch.Tensor) else None
        layer.register_buffer("_vspec_w8a16_bias", quant_bias)
        layer.quant_method = _WeightOnlyLinearMethod()
        configured.append(name)

    if selected is not None:
        missing = selected.difference(available)
        if missing:
            raise RuntimeError("Unknown EAGLE body W8A16 layer(s): " + ", ".join(sorted(missing)))
    if not configured:
        raise RuntimeError("EAGLE body W8A16 found no eligible linear layers")
    logger.info(
        "Enabled online EAGLE body W8A16 for %d layers: %s",
        len(configured),
        ", ".join(configured),
    )


def apply_eagle_body_quantization_patch() -> bool:
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )

    if getattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, False):
        return False
    _install_compiler_compatibility()
    original_load_model = AscendSpecDecodeBaseProposer.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        _configure_body_quantization(self)
        return result

    AscendSpecDecodeBaseProposer.load_model = load_model
    setattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, True)
    return True
