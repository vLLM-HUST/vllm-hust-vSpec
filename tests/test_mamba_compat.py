from __future__ import annotations

import pytest
import torch

from vllm_hust_vspec.backends.mamba_compat import iter_cache_tensors


def test_iter_cache_tensors_flattens_hybrid_planes() -> None:
    first = torch.empty(2)
    second = torch.empty(3)
    third = torch.empty(4)

    planes = list(iter_cache_tensors([first, (second, [third])]))
    assert planes[0] is first
    assert planes[1] is second
    assert planes[2] is third


def test_iter_cache_tensors_rejects_unknown_entries() -> None:
    with pytest.raises(TypeError, match="KV-cache Tensor or plane sequence"):
        list(iter_cache_tensors([object()]))
