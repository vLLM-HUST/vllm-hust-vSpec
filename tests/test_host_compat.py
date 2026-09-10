from __future__ import annotations

import pytest

from vllm_hust_vspec.backends.kv_cache import (
    PATCH_MARKER,
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
