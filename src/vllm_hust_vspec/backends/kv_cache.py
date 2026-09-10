"""Compatibility for vLLM's multi-attention-layer KV-cache guard."""

from __future__ import annotations

from typing import Any

PATCH_MARKER = "_vllm_hust_vspec_multi_layer_kv_cache_patched"


def apply_multi_layer_kv_cache_patch(platform_cls: Any = None) -> bool:
    """Declare support only when Ascend has not provided its own declaration.

    Serial speculative decoding registers Target and Draft attention modules with
    matching layer indices. Ascend binds each module's cache by its full layer
    name, so its runner can retain both entries just like the CUDA runner.
    """
    if platform_cls is None:
        from vllm_ascend.platform import NPUPlatform

        platform_cls = NPUPlatform

    if getattr(platform_cls, PATCH_MARKER, False):
        return False
    if "check_runner_kv_caches_multi_layer" in vars(platform_cls):
        return False

    @classmethod
    def check_runner_kv_caches_multi_layer(cls: Any) -> None:
        return None

    platform_cls.check_runner_kv_caches_multi_layer = check_runner_kv_caches_multi_layer
    setattr(platform_cls, PATCH_MARKER, True)
    return True
