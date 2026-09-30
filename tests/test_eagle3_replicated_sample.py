from __future__ import annotations

import torch

from vllm_hust_vspec.backends.eagle3_replicated_sample import (
    _sample_replicated_draft_logits,
)


def test_sample_replicated_draft_logits_maps_reduced_vocab() -> None:
    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0, -1.0],
            [3.0, 1.0, 4.0, 2.0],
        ]
    )
    mapping_bias = torch.tensor([100, 200, 300, 400])

    result = _sample_replicated_draft_logits(logits, mapping_bias)

    assert result.tolist() == [201, 302]
