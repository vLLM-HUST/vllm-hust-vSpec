# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen2/Qwen2.5 EAGLE draft model supplied out of tree by vSpec."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import Qwen2Config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen2 import (
    Qwen2DecoderLayer as BaseQwen2DecoderLayer,
)
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)


def _zero_optional_attention_biases(model: nn.Module) -> None:
    """Initialize checkpoint-optional QKV biases deterministically."""
    with torch.no_grad():
        for layer in model.layers:
            bias = layer.self_attn.qkv_proj.bias
            if bias is not None:
                bias.zero_()


class Qwen2EagleDecoderLayer(BaseQwen2DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        disable_input_layernorm: bool,
        prefix: str = "",
        config: Qwen2Config | None = None,
    ) -> None:
        assert config is not None
        super().__init__(
            config=config,
            cache_config=vllm_config.cache_config,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=prefix,
        )
        if disable_input_layernorm:
            self.input_layernorm = nn.Identity()


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "batch"},
        "positions": {0: "batch"},
        "hidden_states": {0: "batch"},
    }
)
class Qwen2Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                Qwen2EagleDecoderLayer(
                    vllm_config,
                    index == 0,
                    prefix=maybe_prefix(prefix, f"layers.{index + start_layer_id}"),
                    config=self.config,
                )
                for index in range(self.config.num_hidden_layers)
            ]
        )
        self.fc = ReplicatedLinear(
            input_size=self.config.hidden_size * 2,
            output_size=self.config.hidden_size,
            bias=True,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = hidden_states + residual
        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        _zero_optional_attention_biases(self)
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen2ForCausalLMEagle(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = self.config.vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.config.target_layer_count = target_layer_num
        self.model = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=logit_scale,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multimodal inputs yet."
            )
        return self.model(input_ids, positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def transform(item: tuple[str, torch.Tensor]) -> tuple[str, torch.Tensor]:
            name, loaded_weight = item
            if "lm_head" not in name:
                name = "model." + name
            process_eagle_weight(self, name)
            return name, loaded_weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(map(transform, weights))


EagleQwen2ForCausalLM = Qwen2ForCausalLMEagle
EagleQwen2ForCausalLMEagle = Qwen2ForCausalLMEagle

__all__ = [
    "EagleQwen2ForCausalLM",
    "EagleQwen2ForCausalLMEagle",
    "Qwen2ForCausalLMEagle",
    "Qwen2Model",
]
