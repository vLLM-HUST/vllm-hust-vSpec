"""Compatibility helpers for current vLLM hybrid Mamba cache APIs."""

from __future__ import annotations

from functools import wraps
from typing import Any


def iter_cache_tensors(kv_caches: Any) -> Any:
    """Flatten hybrid cache planes while rejecting unknown containers."""
    import torch

    for cache in kv_caches:
        if isinstance(cache, torch.Tensor):
            yield cache
        elif isinstance(cache, (tuple, list)):
            yield from iter_cache_tensors(cache)
        else:
            raise TypeError(
                f"vSpec expected a KV-cache Tensor or plane sequence, got {type(cache).__name__}"
            )


def _apply_hybrid_cache_copy_patch() -> bool:
    from vllm.v1.worker import gpu_model_runner

    original_copy = gpu_model_runner.copy_kv_cache_blocks_inplace
    if getattr(original_copy, "_vspec_hybrid_cache_planes", False):
        return False

    @wraps(original_copy)
    def copy_kv_cache_blocks_inplace(
        kv_caches: Any,
        num_blocks: int,
        kv_cache_block_copies: Any,
    ) -> None:
        original_copy(
            iter_cache_tensors(kv_caches),
            num_blocks,
            kv_cache_block_copies,
        )

    copy_kv_cache_blocks_inplace._vspec_hybrid_cache_planes = True  # type: ignore[attr-defined]
    gpu_model_runner.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
    return True


def get_current_mamba_groups(kv_cache_config: Any) -> dict[Any, list[int]]:
    """Return the Mamba group mapping expected by current vLLM workers."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    mamba_groups: dict[Any, set[int]] = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, MambaSpec):
            mamba_groups.setdefault(group_spec, set()).add(group_id)
            continue
        if not isinstance(group_spec, UniformTypeKVCacheSpecs):
            continue
        for layer_name in group.layer_names:
            layer_spec = group_spec.kv_cache_specs.get(layer_name)
            if isinstance(layer_spec, MambaSpec):
                mamba_groups.setdefault(layer_spec, set()).add(group_id)

    if not mamba_groups:
        raise RuntimeError("vSpec could not identify a Mamba KV-cache group")
    return {spec: sorted(group_ids) for spec, group_ids in mamba_groups.items()}


def normalize_mamba_state_copy_funcs(
    kv_cache_config: Any,
    copy_funcs: Any,
) -> Any:
    """Map the legacy model copy-function tuple to current Mamba types."""
    if isinstance(copy_funcs, dict):
        return copy_funcs
    if not isinstance(copy_funcs, tuple):
        raise TypeError(
            "vSpec expected Mamba state copy functions as a tuple or mapping, "
            f"got {type(copy_funcs).__name__}"
        )
    return {spec.mamba_type: copy_funcs for spec in get_current_mamba_groups(kv_cache_config)}


def install_current_mamba_runtime_api() -> None:
    """Restore current APIs after vLLM-Ascend installs its worker patches."""
    from vllm.v1.worker import mamba_utils

    mamba_utils.get_mamba_groups = get_current_mamba_groups

    original_postprocess = mamba_utils.postprocess_mamba_align_gpu
    if getattr(original_postprocess, "_vspec_mamba_copy_api", False):
        return
    if not original_postprocess.__module__.startswith("vllm.v1.worker.mamba_utils"):
        return

    @wraps(original_postprocess)
    def postprocess_mamba_align_gpu(*args: Any, **kwargs: Any) -> Any:
        if "mamba_state_copy_funcs" in kwargs:
            kwargs["mamba_state_copy_funcs"] = normalize_mamba_state_copy_funcs(
                kwargs["kv_cache_config"],
                kwargs["mamba_state_copy_funcs"],
            )
        return original_postprocess(*args, **kwargs)

    postprocess_mamba_align_gpu._vspec_mamba_copy_api = True  # type: ignore[attr-defined]
    mamba_utils.postprocess_mamba_align_gpu = postprocess_mamba_align_gpu


def apply_mamba_runtime_compatibility_patch() -> bool:
    """Install current Mamba helpers lazily after late Ascend patching."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    applied = _apply_hybrid_cache_copy_patch()
    original_get_copy_funcs = GPUModelRunner._get_mamba_state_copy_funcs
    if getattr(original_get_copy_funcs, "_vspec_mamba_group_api", False):
        return applied

    @wraps(original_get_copy_funcs)
    def get_mamba_state_copy_funcs(self: Any) -> Any:
        install_current_mamba_runtime_api()
        return original_get_copy_funcs(self)

    get_mamba_state_copy_funcs._vspec_mamba_group_api = True  # type: ignore[attr-defined]
    GPUModelRunner._get_mamba_state_copy_funcs = get_mamba_state_copy_funcs
    return True
