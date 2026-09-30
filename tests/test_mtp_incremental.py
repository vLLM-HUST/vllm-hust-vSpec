from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_hust_vspec.backends.mtp import (
    _is_async_hybrid_mamba_runner,
    _is_async_mtp_mamba_runner,
    _remap_accepted_tokens_on_device,
    _uniform_step3p5_batch_size,
)


def _metadata(query_start_loc: list[int], num_actual_tokens: int, num_reqs: int):
    return SimpleNamespace(
        query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
        num_actual_tokens=num_actual_tokens,
        num_reqs=num_reqs,
    )


def test_uniform_step3p5_batch_size_accepts_graph_padded_batch() -> None:
    metadata = _metadata([0, 3, 6, 9, 12], num_actual_tokens=12, num_reqs=8)

    assert _uniform_step3p5_batch_size(metadata, num_speculative_tokens=2) == 4


def test_uniform_step3p5_batch_size_rejects_mixed_queries() -> None:
    metadata = _metadata([0, 2, 6], num_actual_tokens=6, num_reqs=2)

    assert _uniform_step3p5_batch_size(metadata, num_speculative_tokens=2) is None


def test_uniform_step3p5_batch_size_rejects_partial_window() -> None:
    metadata = _metadata([0, 3, 5], num_actual_tokens=5, num_reqs=2)

    assert _uniform_step3p5_batch_size(metadata, num_speculative_tokens=2) is None


def test_remap_accepted_tokens_on_device_handles_reorder_and_new_rows() -> None:
    source = torch.tensor([3, 1, 2, 3], dtype=torch.int32)
    prev_positions = torch.tensor([2, -1, 0], dtype=torch.int64)
    destination = torch.zeros(5, dtype=torch.int32)
    safe_positions = torch.empty(5, dtype=torch.int64)
    new_request_mask = torch.empty(5, dtype=torch.bool)

    _remap_accepted_tokens_on_device(
        source,
        prev_positions,
        destination,
        safe_positions,
        new_request_mask,
        num_reqs=3,
    )

    assert destination.tolist() == [2, 1, 3, 1, 1]


def test_async_mtp_mamba_runner_gate() -> None:
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
        use_async_scheduling=True,
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
    )

    assert _is_async_mtp_mamba_runner(runner)
    runner.speculative_config.method = "eagle3"
    assert not _is_async_mtp_mamba_runner(runner)
    assert _is_async_hybrid_mamba_runner(runner, frozenset({"eagle3"}))
    assert not _is_async_hybrid_mamba_runner(runner, frozenset({"mtp"}))
