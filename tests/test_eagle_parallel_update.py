from __future__ import annotations

from vllm_hust_vspec.backends.draft_parallel_update import (
    _normalize_dense_fia_param,
)


def test_normalize_current_dense_fia_capture_layout() -> None:
    values = list(range(22))
    values[20] = object()
    values[21] = "model.layers.3.self_attn.attn"

    normalized = _normalize_dense_fia_param(tuple(values))

    assert normalized is not None
    assert normalized.sliding_window == 16
    assert normalized.c8_k_scale == 17
    assert normalized.c8_v_scale == 19
    assert normalized.layer_name == "model.layers.3.self_attn.attn"
    assert normalized.adaptive_decision is None


def test_normalize_legacy_dense_fia_capture_layout() -> None:
    values = list(range(21))
    values[20] = "model.layers.7.self_attn.attn"

    normalized = _normalize_dense_fia_param(tuple(values))

    assert normalized is not None
    assert normalized.sliding_window is None
    assert normalized.c8_k_scale == 16
    assert normalized.c8_v_scale == 18
    assert normalized.layer_name == "model.layers.7.self_attn.attn"
    assert normalized.adaptive_decision is None
