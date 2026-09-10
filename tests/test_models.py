from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch.nn as nn

from vllm_hust_vspec.models import qwen2_eagle, register_models


def test_qwen2_eagle_records_target_layer_count() -> None:
    draft_config = SimpleNamespace(
        draft_vocab_size=None,
        vocab_size=151936,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft_config),
        ),
        model_config=SimpleNamespace(get_num_layers=lambda parallel_config: 48),
        parallel_config=SimpleNamespace(),
    )

    with (
        mock.patch.object(qwen2_eagle, "Qwen2Model", return_value=nn.Identity()),
        mock.patch.object(qwen2_eagle, "LogitsProcessor", return_value=nn.Identity()),
    ):
        model = qwen2_eagle.Qwen2ForCausalLMEagle(vllm_config=vllm_config)

    assert model.config.target_layer_count == 48
    assert model.config.draft_vocab_size == model.config.vocab_size


def test_qwen2_eagle_is_importable_through_model_registry() -> None:
    from vllm import ModelRegistry

    register_models()
    model_info = ModelRegistry._try_inspect_model_cls("EagleQwen2ForCausalLM")

    assert model_info is not None
