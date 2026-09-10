"""Dispatch vSpec patches to the selected speculative backend."""

from __future__ import annotations

from .config import PluginSettings


def apply_patches(settings: PluginSettings) -> bool:
    from .compatibility import require_host_compatibility

    require_host_compatibility(
        settings.method,
        adaptive=settings.adaptive_speculation,
    )
    from .backends.kv_cache import apply_multi_layer_kv_cache_patch

    applied = apply_multi_layer_kv_cache_patch()
    if settings.adaptive_speculation:
        from .adaptive.runtime import apply_adaptive_patches

        applied = apply_adaptive_patches(settings)
    if settings.method == "draft_model":
        from .backends.draft import apply_draft_patches

        return apply_draft_patches(settings) or applied
    if settings.method in {"eagle", "eagle3"}:
        from .backends.eagle import apply_eagle_patches

        return apply_eagle_patches(settings) or applied
    if settings.method == "dflash":
        # DFlash is provided by vLLM-Ascend; vSpec contributes the adaptive
        # control and graph-width hooks installed above.
        return applied
    raise ValueError(f"unsupported vSpec method: {settings.method}")
