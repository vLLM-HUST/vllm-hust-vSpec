from __future__ import annotations

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_hust_vspec.backends.eagle3_hybrid import (
    _build_compact_group_layer_names,
)


def test_qwen35_eagle3_compact_groups_preserve_small_draft_blocks() -> None:
    page_size = 2_138_112
    target = FullAttentionSpec(
        block_size=2048,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    draft = FullAttentionSpec(
        block_size=128,
        num_kv_heads=16,
        head_size=256,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    mamba = MambaSpec(
        block_size=2048,
        shapes=((10_240,), (262_144,)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=page_size,
    )
    specs = {
        **{
            f"language_model.model.layers.{4 * index + 3}.self_attn.attn": target
            for index in range(10)
        },
        "model.layers.40.self_attn.attn": draft,
        **{f"language_model.model.layers.{index}.linear_attn": mamba for index in range(30)},
    }

    groups = _build_compact_group_layer_names(specs, group_size=4)

    assert groups is not None
    assert [len(group) for group in groups] == [4, 3, 3, 4, 4, 4, 4, 4, 4, 3, 3, 1]
    assert sorted(name for group in groups for name in group) == sorted(specs)
