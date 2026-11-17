"""Narrow KV-binding bridge for a Qwen3.5 target and one MTP head.

The generic vLLM runner rejects duplicate numeric attention-layer indices.
For this model the target and the one MTP head may both have index zero, while
the Ascend runner binds their caches by full name. Only this exact pair is
allowed to bypass the generic guard; all other duplicate indices still use
the original check.
"""

from __future__ import annotations

import contextvars
import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

_allow_expected_pair: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "qwen35_mtp_expected_kv_pair", default=False
)
_installed = False


def _duplicate_indices(kv_caches: dict[str, Any], num_attn_module: int) -> dict[int, list[str]]:
    from vllm.model_executor.models.utils import extract_layer_index

    by_index: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        by_index[extract_layer_index(layer_name, num_attn_module)].append(layer_name)
    return {index: names for index, names in by_index.items() if len(names) > 1}


def _expected_qwen35_mtp_pair(kv_caches: dict[str, Any], num_attn_module: int) -> bool:
    duplicates = _duplicate_indices(kv_caches, num_attn_module)
    if set(duplicates) != {0} or len(duplicates[0]) != 2:
        return False
    # Ascend's proposer may load without the draft_model prefix. In either
    # case exactly one cache must belong to the single MTP attention head.
    draft_names = [
        name
        for name in duplicates[0]
        if name.startswith(("mtp.layers.0.", "draft_model.mtp.layers.0."))
    ]
    if len(draft_names) != 1:
        return False
    draft = draft_names[0]
    target = next(name for name in duplicates[0] if name != draft)
    return (
        not target.startswith(("draft_model.", "mtp."))
        and ".layers.0." in f".{target}"
        and kv_caches[draft] is not kv_caches[target]
    )


def install() -> bool:
    """Install before worker imports bind_kv_cache; idempotent per process."""
    global _installed
    if _installed:
        return False

    from vllm.v1.worker import utils as worker_utils
    from vllm_ascend.platform import NPUPlatform

    if getattr(NPUPlatform, "_vllm_hust_vspec_multi_layer_kv_cache_patched", False):
        raise RuntimeError("Enable Qwen3.5 MTP compatibility before vSpec plugin registration")

    original_bind = worker_utils.bind_kv_cache
    original_check = NPUPlatform.check_runner_kv_caches_multi_layer

    @classmethod
    def checked_multi_layer(cls: type[Any]) -> None:
        if _allow_expected_pair.get():
            return None
        return original_check()

    def checked_bind_kv_cache(
        kv_caches: dict[str, Any],
        forward_context: dict[str, Any],
        runner_kv_caches: list[Any],
        num_attn_module: int = 1,
        kv_cache_groups: Sequence[Any] | None = None,
    ) -> None:
        allow = _expected_qwen35_mtp_pair(kv_caches, num_attn_module)
        if not allow:
            duplicates = _duplicate_indices(kv_caches, num_attn_module)
            if duplicates:
                logging.getLogger(__name__).error(
                    "Qwen3.5 MTP KV guard rejected duplicate layer indices: %s",
                    duplicates,
                )
        token = _allow_expected_pair.set(allow)
        try:
            return original_bind(
                kv_caches,
                forward_context,
                runner_kv_caches,
                num_attn_module,
                kv_cache_groups,
            )
        finally:
            _allow_expected_pair.reset(token)

    checked_bind_kv_cache._qwen35_mtp_kv_guard = True  # type: ignore[attr-defined]
    NPUPlatform.check_runner_kv_caches_multi_layer = checked_multi_layer
    worker_utils.bind_kv_cache = checked_bind_kv_cache
    _installed = True
    return True
