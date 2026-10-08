from __future__ import annotations

import pytest

from vllm_hust_vspec.backends.kv_cache import (
    PAGE_SIZE_PATCH_MARKER,
    PATCH_MARKER,
    align_eagle_kv_cache_page_size,
    apply_eagle_kv_cache_page_size_patch,
    apply_multi_layer_kv_cache_patch,
)
from vllm_hust_vspec.host_compat import call_with_supported_kwargs


def test_call_with_supported_kwargs_filters_removed_host_argument() -> None:
    def host(value: int, *, retained: int) -> int:
        return value + retained

    assert (
        call_with_supported_kwargs(
            host,
            2,
            retained=3,
            removed=True,
        )
        == 5
    )


def test_call_with_supported_kwargs_preserves_variadic_host_arguments() -> None:
    def host(**kwargs):
        return kwargs

    assert call_with_supported_kwargs(host, current=1, legacy=2) == {
        "current": 1,
        "legacy": 2,
    }


def test_multi_layer_kv_cache_patch_replaces_inherited_guard() -> None:
    class Platform:
        @classmethod
        def check_runner_kv_caches_multi_layer(cls) -> None:
            raise NotImplementedError

    class NPUPlatform(Platform):
        pass

    assert apply_multi_layer_kv_cache_patch(NPUPlatform)
    NPUPlatform.check_runner_kv_caches_multi_layer()
    assert getattr(NPUPlatform, PATCH_MARKER)
    assert not apply_multi_layer_kv_cache_patch(NPUPlatform)


def test_multi_layer_kv_cache_patch_preserves_native_override() -> None:
    class NPUPlatform:
        @classmethod
        def check_runner_kv_caches_multi_layer(cls) -> None:
            raise RuntimeError("native declaration")

    assert not apply_multi_layer_kv_cache_patch(NPUPlatform)
    with pytest.raises(RuntimeError, match="native declaration"):
        NPUPlatform.check_runner_kv_caches_multi_layer()
    assert not hasattr(NPUPlatform, PATCH_MARKER)


def test_eagle_page_alignment_uses_rank_local_kv_heads() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FakeAttentionSpec:
        block_size: int
        num_kv_heads: int
        token_bytes_per_head: int
        page_size_padded: int | None = None
        indexes_kv_by_block_stride: bool = False

        @property
        def real_page_size_bytes(self) -> int:
            return self.block_size * self.num_kv_heads * self.token_bytes_per_head

        @property
        def page_size_bytes(self) -> int:
            return self.page_size_padded or self.real_page_size_bytes

    @dataclass(frozen=True)
    class FakeMambaSpec:
        page_size_bytes: int

    specs = {
        "target": FakeAttentionSpec(261, 8, 1024),
        "state": FakeMambaSpec(2_138_112),
        "draft": FakeAttentionSpec(128, 16, 1024),
        "draft.1": FakeAttentionSpec(128, 16, 1024),
    }

    assert align_eagle_kv_cache_page_size(
        specs,
        draft_tensor_parallel_size=2,
        attention_spec_cls=FakeAttentionSpec,
        mamba_spec_cls=FakeMambaSpec,
    )
    assert specs["draft"].num_kv_heads == 8
    assert specs["draft"].block_size == 128
    assert specs["draft"].page_size_bytes == 2_138_112
    assert specs["draft"].indexes_kv_by_block_stride
    assert specs["draft.1"] == specs["draft"]


def test_eagle_page_size_patch_wraps_grouping_once(monkeypatch) -> None:
    from types import SimpleNamespace

    import vllm_hust_vspec.backends.kv_cache as kv_cache

    calls = []
    align_calls = []
    module = SimpleNamespace(
        get_kv_cache_groups=lambda config, specs: calls.append((config, specs)) or []
    )
    monkeypatch.setattr(
        kv_cache,
        "align_eagle_kv_cache_page_size",
        lambda specs, **kwargs: align_calls.append((specs, kwargs)) or True,
    )
    assert apply_eagle_kv_cache_page_size_patch(module)
    assert not apply_eagle_kv_cache_page_size_patch(module)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_tensor_parallel_size=2),
        cache_config=SimpleNamespace(block_size=128),
    )
    assert module.get_kv_cache_groups(config, {}) == []
    assert calls == [(config, {})]
    assert align_calls == [({}, {"draft_tensor_parallel_size": 2})]
    assert getattr(module, PAGE_SIZE_PATCH_MARKER)
