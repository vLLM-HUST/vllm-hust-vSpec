"""Compatibility for vLLM's multi-attention-layer KV-cache guard."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from typing import Any

PATCH_MARKER = "_vllm_hust_vspec_multi_layer_kv_cache_patched"
PAGE_SIZE_PATCH_MARKER = "_vllm_hust_vspec_eagle_page_size_patched"
logger = logging.getLogger(__name__)


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


def align_eagle_kv_cache_page_size(
    kv_cache_specs: dict[str, Any],
    *,
    draft_tensor_parallel_size: int,
    kernel_block_size: int = 128,
    attention_spec_cls: type[Any] | None = None,
    mamba_spec_cls: type[Any] | None = None,
) -> bool:
    """Align an EAGLE attention page with a hybrid target's physical page.

    Qwen3.5 hybrid targets pad their full-attention cache to the recurrent-state
    page size.  vLLM merges the EAGLE cache into the same allocation plan, but
    the draft spec still carries global KV-head count and its ordinary block
    size.  Convert that one outlier to rank-local heads and choose the exact
    logical block size that matches the target page.
    """
    if draft_tensor_parallel_size < 1:
        raise ValueError("draft tensor parallel size must be positive")

    if attention_spec_cls is None or mamba_spec_cls is None:
        try:
            from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
        except ImportError:
            return False
        attention_spec_cls = AttentionSpec
        mamba_spec_cls = MambaSpec

    if not any(isinstance(spec, mamba_spec_cls) for spec in kv_cache_specs.values()):
        return False

    page_counts = Counter(spec.page_size_bytes for spec in kv_cache_specs.values())
    target_page_size, target_count = page_counts.most_common(1)[0]
    logger.warning(
        "vSpec KV page candidates: %s",
        [
            (
                name,
                type(spec).__name__,
                getattr(spec, "block_size", None),
                getattr(spec, "num_kv_heads", None),
                spec.page_size_bytes,
                getattr(spec, "page_size_padded", None),
            )
            for name, spec in kv_cache_specs.items()
        ],
    )
    if target_count < 2:
        return False

    candidates = [
        (name, spec)
        for name, spec in kv_cache_specs.items()
        if isinstance(spec, attention_spec_cls)
        and spec.page_size_bytes != target_page_size
        and getattr(spec, "page_size_padded", None) is None
    ]
    if not candidates:
        return False
    if kernel_block_size < 1:
        raise ValueError("kernel block size must be positive")
    aligned_specs: dict[str, Any] = {}
    for name, spec in candidates:
        global_kv_heads = int(spec.num_kv_heads)
        if global_kv_heads % draft_tensor_parallel_size:
            return False
        local_kv_heads = global_kv_heads // draft_tensor_parallel_size
        bytes_per_token_per_head = spec.real_page_size_bytes // (
            int(spec.block_size) * global_kv_heads
        )
        real_page_size = bytes_per_token_per_head * local_kv_heads * kernel_block_size
        if bytes_per_token_per_head <= 0 or real_page_size > target_page_size:
            return False

        aligned = replace(
            spec,
            block_size=kernel_block_size,
            num_kv_heads=local_kv_heads,
            page_size_padded=target_page_size,
            indexes_kv_by_block_stride=True,
        )
        if aligned.page_size_bytes != target_page_size:
            return False
        aligned_specs[name] = aligned

    kv_cache_specs.update(aligned_specs)
    logger.warning("Aligned %d EAGLE KV specs to %d bytes", len(aligned_specs), target_page_size)
    return True


def apply_eagle_kv_cache_page_size_patch(kv_cache_utils: Any = None) -> bool:
    """Patch the cache-group boundary where target and EAGLE specs meet."""
    if kv_cache_utils is None:
        from vllm.v1.core import kv_cache_utils

    if getattr(kv_cache_utils, PAGE_SIZE_PATCH_MARKER, False):
        return False

    original = kv_cache_utils.get_kv_cache_groups

    def get_kv_cache_groups(vllm_config: Any, kv_cache_specs: dict[str, Any]):
        speculative = getattr(vllm_config, "speculative_config", None)
        draft_tp = int(getattr(speculative, "draft_tensor_parallel_size", 1) or 1)
        align_eagle_kv_cache_page_size(
            kv_cache_specs,
            draft_tensor_parallel_size=draft_tp,
        )
        return original(vllm_config, kv_cache_specs)

    kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
    setattr(kv_cache_utils, PAGE_SIZE_PATCH_MARKER, True)
    return True
