"""EAGLE3 compatibility for hybrid Attention/Mamba KV-cache pools."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import replace
from functools import wraps
from typing import Any

import torch

PATCH_MARKER = "_vllm_hust_vspec_eagle3_hybrid_patched"
ZEROER_PATCH_MARKER = "_vllm_hust_vspec_eagle3_zeroer_patched"
DRAFT_BLOCK_PATCH_MARKER = "_vllm_hust_vspec_eagle3_draft_block_patched"
GROUP_ANNOTATION_PATCH_MARKER = "_vllm_hust_vspec_eagle3_group_annotation_patched"
COMPACT_GROUP_PATCH_MARKER = "_vllm_hust_vspec_eagle3_compact_group_patched"
COHORT_REFILL_PATCH_MARKER = "_vllm_hust_vspec_eagle3_cohort_refill_patched"
FUSED_PRECOPY_PATCH_MARKER = "_vllm_hust_vspec_eagle3_fused_precopy_patched"
_BOUNDARY_ERROR = "Aliased attention/Mamba pools disagree on native plane boundaries"

_QWEN35_TARGET_LAYER_PREFIX = "language_model.model.layers."
_QWEN35_DRAFT_LAYER_PREFIX = "model.layers."


def _build_compact_group_layer_names(
    kv_cache_spec: dict[str, Any],
    group_size: int,
) -> list[list[str]] | None:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    target_names = [
        name
        for name, spec in kv_cache_spec.items()
        if name.startswith(_QWEN35_TARGET_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    ]
    draft_names = [
        name
        for name, spec in kv_cache_spec.items()
        if name.startswith(_QWEN35_DRAFT_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    ]
    mamba_names = [
        name
        for name, spec in kv_cache_spec.items()
        if name.startswith(_QWEN35_TARGET_LAYER_PREFIX) and isinstance(spec, MambaSpec)
    ]
    if (
        not target_names
        or not draft_names
        or not mamba_names
        or len(target_names) + len(draft_names) + len(mamba_names) != len(kv_cache_spec)
    ):
        return None

    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    target_block_sizes = {kv_cache_spec[name].block_size for name in target_names + mamba_names}
    draft_block_sizes = {kv_cache_spec[name].block_size for name in draft_names}
    if len(page_sizes) != 1 or len(target_block_sizes) != 1 or len(draft_block_sizes) != 1:
        return None
    if group_size <= 0:
        raise ValueError("VSPEC_EAGLE3_KV_GROUP_SIZE must be positive")

    def split(names: list[str]) -> list[list[str]]:
        num_groups = math.ceil(len(names) / group_size)
        return [names[index::num_groups] for index in range(num_groups)]

    return [*split(target_names), *split(mamba_names), *split(draft_names)]


def apply_eagle3_cohort_refill_patch() -> bool:
    """Select EAGLE3 replacement admission from measured online goodput."""
    from ..adaptive.refill import install_online_refill_patch

    return install_online_refill_patch(
        method="eagle3",
        enabled_env="HUST_VSPEC_EAGLE3_COHORT_REFILL",
        patch_marker=COHORT_REFILL_PATCH_MARKER,
    )


def apply_eagle3_fused_mamba_precopy_patch() -> bool:
    """Pass the align context omitted by the Ascend hybrid runner.

    The shared vLLM Mamba helper already provides a device-side, all-group
    pre-copy kernel.  The generic runner supplies ``align_ctx`` to select it,
    while the current Ascend runner calls the helper without that argument.
    Keep the existing accepted-count synchronization semantics and only
    restore the fused state-copy path here.
    """
    if os.getenv("HUST_VSPEC_EAGLE3_FUSED_PRECOPY", "0") == "0":
        return False

    from vllm.logger import logger
    from vllm.v1.worker import mamba_utils
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_prepare_inputs = NPUModelRunner._prepare_inputs
    if getattr(original_prepare_inputs, FUSED_PRECOPY_PATCH_MARKER, False):
        return False
    original_preprocess_mamba = mamba_utils.preprocess_mamba

    @wraps(original_prepare_inputs)
    def prepare_inputs(self: Any, *args: Any, **kwargs: Any) -> Any:
        speculative_config = getattr(self, "speculative_config", None)
        if (
            getattr(speculative_config, "method", None) == "eagle3"
            and getattr(getattr(self, "cache_config", None), "mamba_cache_mode", None) == "align"
        ):
            mamba_bufs = self._get_mamba_bufs()
            if mamba_bufs.postprocess_align is not None:
                mamba_bufs.preprocess._vspec_eagle3_align_ctx = mamba_bufs.postprocess_align
        return original_prepare_inputs(self, *args, **kwargs)

    @wraps(original_preprocess_mamba)
    def preprocess_mamba(*args: Any, **kwargs: Any) -> Any:
        copy_bufs = kwargs.get("copy_bufs")
        if copy_bufs is None and len(args) >= 9:
            copy_bufs = args[8]
        align_ctx = getattr(copy_bufs, "_vspec_eagle3_align_ctx", None)
        if align_ctx is not None and kwargs.get("align_ctx") is None:
            kwargs["align_ctx"] = align_ctx
        return original_preprocess_mamba(*args, **kwargs)

    setattr(prepare_inputs, FUSED_PRECOPY_PATCH_MARKER, True)
    setattr(preprocess_mamba, FUSED_PRECOPY_PATCH_MARKER, True)
    NPUModelRunner._prepare_inputs = prepare_inputs
    mamba_utils.preprocess_mamba = preprocess_mamba
    logger.info("vSpec enabled fused Mamba pre-copy for EAGLE3 hybrid batches")
    return True


def _allocate_active_page_hybrid_cache(
    config: Any,
    device: torch.device,
    layout: Any,
    kernel_block_sizes: list[int],
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Build native planes inside padded layer slots without using the padding.

    A wide EAGLE3 draft page can increase ``page_size_bytes`` for every layer
    in a hybrid target.  The target Attention and Mamba states still have the
    same unpadded size and may safely alias, while treating the padding as
    state changes their plane boundaries and rejects an otherwise valid pool.
    """
    from vllm.utils.torch_utils import get_dtype_size
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    if layout != KVCacheLayout.LBHNC:
        raise ValueError("EAGLE3 hybrid state planes require layer-compact LBHNC pool placement")

    specs: dict[str, tuple[Any, int]] = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            specs[name] = (spec, kernel_block_sizes[group_id])

    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Hybrid cache descriptors must share one backing allocation")

    blocks = config.num_blocks
    pool_offsets: dict[tuple[int, int, int], set[int]] = {}
    layer_pages: dict[str, tuple[int, int, int]] = {}
    layer_original_offsets: dict[str, int] = {}
    for tensor in config.kv_cache_tensors:
        for index, name in enumerate(tensor.layers):
            spec, kernel_size = specs[name]
            allocated_page = spec.page_size_bytes
            offset = tensor.offset + index * tensor.layer_stride
            if (
                tensor.block_stride != allocated_page
                or tensor.layer_stride < blocks * allocated_page
            ):
                raise ValueError("EAGLE3 active-page cache requires non-interleaved layer pools")
            if offset < 0 or offset + blocks * allocated_page > tensor.size:
                raise ValueError("Hybrid layer pool lies outside its backing allocation")

            if isinstance(spec, MambaSpec):
                planes = tuple(
                    math.prod(shape) * get_dtype_size(dtype)
                    for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
                )
                if len(planes) not in (1, 2):
                    raise ValueError("EAGLE3 hybrid cache supports one or two Mamba state planes")
                prefix, state = (0, planes[0]) if len(planes) == 1 else planes
                active_page = sum(planes)
            elif isinstance(spec, AttentionSpec):
                if spec.head_size != spec.head_size_v or spec.block_size % kernel_size:
                    raise ValueError(
                        "EAGLE3 hybrid cache requires equal K/V head sizes and "
                        "divisible kernel blocks"
                    )
                state = (
                    spec.block_size
                    * spec.num_kv_heads
                    * spec.head_size
                    * get_dtype_size(spec.dtype)
                )
                active_page = spec.unpadded_page_size_bytes
                prefix = active_page - 2 * state
            else:
                raise ValueError(f"Unsupported EAGLE3 hybrid cache spec: {type(spec).__name__}")

            if active_page > allocated_page or prefix < 0 or prefix + state > active_page:
                raise ValueError("Native state planes exceed the active cache page")
            boundaries = (prefix, state, active_page)
            pool_offsets.setdefault(boundaries, set()).add(offset)
            layer_pages[name] = boundaries
            layer_original_offsets[name] = offset

    compact_offsets: dict[tuple[tuple[int, int, int], int], int] = {}
    backing_size = 0
    for boundaries, original_offsets in sorted(pool_offsets.items()):
        active_page = boundaries[2]
        for original_offset in sorted(original_offsets):
            compact_offsets[(boundaries, original_offset)] = backing_size
            backing_size += blocks * active_page

    reserved_size = sizes.pop()
    if backing_size > reserved_size:
        extra_percent = (backing_size / reserved_size - 1) * 100
        from vllm.logger import logger

        logger.info(
            "vSpec EAGLE3: boundary-safe KV pools require %.2f%% more "
            "physical memory than the aliased cache descriptor",
            extra_percent,
        )
    backing = torch.zeros(backing_size, dtype=torch.int8, device=device)
    result: dict[str, tuple[torch.Tensor, ...]] = {}
    for tensor in config.kv_cache_tensors:
        for name in tensor.layers:
            spec, kernel_size = specs[name]
            prefix, state, active_page = layer_pages[name]
            original_offset = layer_original_offsets[name]
            offset = compact_offsets[((prefix, state, active_page), original_offset)]
            raw = backing.narrow(0, offset, blocks * active_page)
            if isinstance(spec, MambaSpec):
                views, start = [], 0
                for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True):
                    size = blocks * math.prod(shape) * get_dtype_size(dtype)
                    views.append(raw.narrow(0, start, size).view(dtype).view(blocks, *shape))
                    start += size
                result[name] = tuple(views)
            else:
                shape = (
                    blocks * spec.block_size // kernel_size,
                    kernel_size,
                    spec.num_kv_heads,
                    spec.head_size,
                )
                result[name] = tuple(
                    raw.narrow(
                        0,
                        blocks * (prefix + plane * state),
                        blocks * state,
                    )
                    .view(spec.dtype)
                    .view(shape)
                    for plane in range(2)
                )
    return result


def apply_eagle3_hybrid_cache_patch() -> bool:
    """Retry only padded-page boundary failures with active-page views."""
    from vllm.logger import logger
    from vllm_ascend.worker import hybrid_cache

    if getattr(hybrid_cache, PATCH_MARKER, False):
        return False

    original = hybrid_cache.allocate_native_hybrid_cache

    def allocate_native_hybrid_cache(
        config: Any,
        device: torch.device,
        layout: Any,
        kernel_block_sizes: list[int],
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        try:
            return original(config, device, layout, kernel_block_sizes)
        except ValueError as exc:
            if str(exc) != _BOUNDARY_ERROR:
                raise
            logger.info(
                "vSpec EAGLE3: using active-page Attention/Mamba KV views inside padded cache slots"
            )
            return _allocate_active_page_hybrid_cache(config, device, layout, kernel_block_sizes)

    hybrid_cache.allocate_native_hybrid_cache = allocate_native_hybrid_cache
    setattr(hybrid_cache, PATCH_MARKER, True)
    return True


def apply_eagle3_hybrid_group_annotation_patch() -> bool:
    """Mark only the EAGLE3 draft cache group in a Qwen3.5 hybrid model."""
    from vllm.logger import logger
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils._annotate_eagle_groups
    if getattr(original, GROUP_ANNOTATION_PATCH_MARKER, False):
        return False

    @wraps(original)
    def annotate_eagle_groups(
        vllm_config: Any,
        kv_cache_spec: dict[str, Any],
        kv_cache_groups: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        original(vllm_config, kv_cache_spec, kv_cache_groups, *args, **kwargs)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if getattr(speculative_config, "method", None) != "eagle3":
            return
        model_config = getattr(vllm_config, "model_config", None)
        model_types = {
            getattr(getattr(model_config, "hf_config", None), "model_type", ""),
            getattr(getattr(model_config, "hf_text_config", None), "model_type", ""),
        }
        if not any(str(model_type).startswith("qwen3_5") for model_type in model_types):
            return

        draft_layers = {name for name in kv_cache_spec if name.startswith("model.layers.")}
        if not draft_layers:
            raise RuntimeError("vSpec could not identify the Qwen3.5 EAGLE3 KV-cache layer")
        annotated = []
        for group_id, group in enumerate(kv_cache_groups):
            if draft_layers.intersection(group.layer_names):
                group.is_eagle_group = True
                annotated.append(group_id)
        if not annotated:
            raise RuntimeError(
                "vSpec found Qwen3.5 EAGLE3 layers but none belonged to a KV-cache group"
            )
        if os.getenv("HUST_VSPEC_EAGLE3_GROUP_TRACE", "0") == "1":
            group_layout = []
            for group_id, group in enumerate(kv_cache_groups):
                signatures = sorted(
                    {
                        (
                            type(kv_cache_spec[name]).__name__,
                            kv_cache_spec[name].block_size,
                            kv_cache_spec[name].page_size_bytes,
                        )
                        for name in group.layer_names
                    }
                )
                group_layout.append(
                    (group_id, len(group.layer_names), signatures, group.layer_names)
                )
            logger.info("vSpec Qwen3.5 EAGLE3 KV-cache layout: %s", group_layout)
        logger.info("vSpec marked Qwen3.5 EAGLE3 KV-cache group(s) %s", annotated)

    setattr(annotate_eagle_groups, GROUP_ANNOTATION_PATCH_MARKER, True)
    kv_cache_utils._annotate_eagle_groups = annotate_eagle_groups
    return True


def apply_eagle3_compact_group_patch() -> bool:
    """Avoid one KV group per Qwen3.5 layer when EAGLE3 uses small blocks."""
    from vllm.logger import logger
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils._get_kv_cache_groups_uniform_page_size
    if getattr(original, COMPACT_GROUP_PATCH_MARKER, False):
        return False

    def compact_groups(kv_cache_spec: dict[str, Any]) -> list[Any]:
        group_size = int(os.getenv("VSPEC_EAGLE3_KV_GROUP_SIZE", "4"))
        grouped_names = _build_compact_group_layer_names(
            kv_cache_spec,
            group_size,
        )
        if grouped_names is None:
            return original(kv_cache_spec)
        groups = kv_cache_utils.create_kv_cache_group_specs(
            kv_cache_spec,
            grouped_names,
        )
        target_count = sum(
            name.startswith(_QWEN35_TARGET_LAYER_PREFIX) and "self_attn" in name
            for name in kv_cache_spec
        )
        mamba_count = sum("linear_attn" in name for name in kv_cache_spec)
        logger.info(
            "vSpec Qwen3.5 EAGLE3 compact KV grouping: %d target, "
            "%d Mamba, %d draft layers -> group sizes %s (limit=%d)",
            target_count,
            mamba_count,
            len(kv_cache_spec) - target_count - mamba_count,
            [len(group) for group in grouped_names],
            group_size,
        )
        return groups

    setattr(compact_groups, COMPACT_GROUP_PATCH_MARKER, True)
    kv_cache_utils._get_kv_cache_groups_uniform_page_size = compact_groups
    return True


def apply_eagle3_nonuniform_zeroer_patch() -> bool:
    """Build one KV-block zeroing address table per Attention page size."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm_ascend.worker.utils import AscendKVBlockZeroer

    original_init = AscendKVBlockZeroer.init_meta
    if getattr(original_init, ZEROER_PATCH_MARKER, False):
        return False
    original_zero = AscendKVBlockZeroer.zero_block_ids

    @wraps(original_init)
    def init_meta(
        self: Any,
        attn_groups_iter: Any,
        kernel_block_sizes: list[list[int]],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        groups_by_page: dict[int, list[Any]] = defaultdict(list)
        passthrough_groups: list[Any] = []
        all_groups = list(attn_groups_iter)
        for group in all_groups:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec) or group.kv_cache_group_id >= len(
                kernel_block_sizes
            ):
                passthrough_groups.append(group)
                continue

            kernel_bs = kernel_block_sizes[group.kv_cache_group_id][0]
            ratio = spec.block_size // kernel_bs
            page_sizes = set()
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                for kv in static_forward_context[layer_name].kv_cache:
                    cur_bytes = kv.stride(0) * kv.element_size()
                    if cur_bytes % 4:
                        raise ValueError("EAGLE3 KV zeroing requires 4-byte alignment")
                    page_sizes.add(cur_bytes // 4 * ratio)
            if not page_sizes:
                passthrough_groups.append(group)
                continue
            if len(page_sizes) != 1:
                raise ValueError(
                    "EAGLE3 KV zeroing found multiple page sizes inside one "
                    f"attention group: {sorted(page_sizes)}"
                )
            groups_by_page[page_sizes.pop()].append(group)

        if len(groups_by_page) <= 1:
            return original_init(
                self,
                all_groups,
                kernel_block_sizes,
                cache_dtype,
                runner_only_attn_layers,
                static_forward_context,
            )

        zeroers = []
        for groups in groups_by_page.values():
            zeroer = type(self)(self.device, self.pin_memory)
            original_init(
                zeroer,
                groups,
                kernel_block_sizes,
                cache_dtype,
                runner_only_attn_layers,
                static_forward_context,
            )
            zeroers.append(zeroer)
        if passthrough_groups:
            zeroer = type(self)(self.device, self.pin_memory)
            original_init(
                zeroer,
                passthrough_groups,
                kernel_block_sizes,
                cache_dtype,
                runner_only_attn_layers,
                static_forward_context,
            )
            if zeroer._meta is not None:
                zeroers.append(zeroer)
        self._vspec_page_zeroers = zeroers
        self._meta = None

    @wraps(original_zero)
    def zero_block_ids(self: Any, block_ids: list[int]) -> None:
        zeroers = getattr(self, "_vspec_page_zeroers", None)
        if zeroers is None:
            return original_zero(self, block_ids)
        for zeroer in zeroers:
            original_zero(zeroer, block_ids)

    setattr(init_meta, ZEROER_PATCH_MARKER, True)
    AscendKVBlockZeroer.init_meta = init_meta
    AscendKVBlockZeroer.zero_block_ids = zero_block_ids
    return True


def apply_eagle3_draft_block_size_patch() -> bool:
    """Decouple the wide EAGLE3 draft KV page from target Mamba alignment."""
    from vllm.logger import logger
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner.get_kv_cache_spec
    if getattr(original, DRAFT_BLOCK_PATCH_MARKER, False):
        return False

    @wraps(original)
    def get_kv_cache_spec(self: Any) -> dict[str, Any]:
        specs = original(self)
        requested = int(os.environ.get("VSPEC_EAGLE3_DRAFT_BLOCK_SIZE", "256"))
        host_page_sizes = [
            spec.page_size_bytes
            for name, spec in specs.items()
            if not name.startswith(_QWEN35_DRAFT_LAYER_PREFIX)
        ]
        aligned_page_size = max(host_page_sizes, default=0)
        changed = []
        for name, spec in list(specs.items()):
            if not name.startswith(_QWEN35_DRAFT_LAYER_PREFIX) or not isinstance(
                spec, FullAttentionSpec
            ):
                continue
            if requested <= 0 or spec.block_size % requested:
                raise ValueError(
                    "VSPEC_EAGLE3_DRAFT_BLOCK_SIZE must be a positive divisor "
                    f"of the host block size {spec.block_size}, got {requested}"
                )
            if requested == spec.block_size:
                continue
            specs[name] = replace(
                spec,
                block_size=requested,
                page_size_padded=aligned_page_size or None,
            )
            if specs[name].page_size_bytes < specs[name].unpadded_page_size_bytes:
                raise ValueError("EAGLE3 draft KV page padding is smaller than its active page")
            changed.append((name, spec.block_size, requested))
        if changed:
            logger.info(
                "vSpec EAGLE3: using independent draft KV block size %d "
                "instead of target-aligned size %d for %d layer(s)",
                changed[0][2],
                changed[0][1],
                len(changed),
            )
        return specs

    setattr(get_kv_cache_spec, DRAFT_BLOCK_PATCH_MARKER, True)
    NPUModelRunner.get_kv_cache_spec = get_kv_cache_spec
    return True
