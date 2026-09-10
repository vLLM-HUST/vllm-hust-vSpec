"""Reusable EAGLE speculative metadata for uniform async batches."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import numpy as np

PATCH_MARKER = "_vllm_hust_vspec_metadata_cache_patched"
ENVIRONMENT_NAME = "VLLM_ASCEND_EAGLE_SPEC_METADATA_CACHE"


def apply_metadata_cache_patch() -> bool:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    return _patch_metadata_cache(NPUModelRunner)


def _patch_metadata_cache(model_runner_cls: Any) -> bool:
    """Install the cache across old and current Ascend metadata signatures."""

    if getattr(model_runner_cls, PATCH_MARKER, False):
        return False
    try:
        native_source = inspect.getsource(model_runner_cls._calc_spec_decode_metadata)
    except (OSError, TypeError):
        native_source = ""
    if ENVIRONMENT_NAME in native_source:
        setattr(model_runner_cls, PATCH_MARKER, True)
        return False

    original_calculate = model_runner_cls._calc_spec_decode_metadata

    def calculate(
        self: Any,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        optional_args_cacheable = all(value is None for value in args) and all(
            value is None for value in kwargs.values()
        )
        cache_enabled = (
            self.use_async_scheduling
            and self.speculative_config is not None
            and self.speculative_config.method == "eagle"
            and getattr(self, "pcp_size", 1) * getattr(self, "dcp_size", 1) == 1
            and len(num_draft_tokens) == self.max_num_reqs
            and np.all(num_draft_tokens == self.num_spec_tokens)
            and np.array_equal(
                cu_num_scheduled_tokens,
                self.arange_np[1 : self.max_num_reqs + 1] * (self.num_spec_tokens + 1),
            )
            and optional_args_cacheable
        )
        cache_key = (
            num_draft_tokens.tobytes(),
            cu_num_scheduled_tokens.tobytes(),
        )
        cached = getattr(self, "_vspec_eagle_metadata_cache", None)
        if cache_enabled and cached is not None and cached[0] == cache_key:
            metadata = cached[1]
            draft_token_ids = self.input_ids.gpu[metadata.logits_indices]
            draft_token_ids = draft_token_ids[metadata.target_logits_indices + 1]
            return replace(metadata, draft_token_ids=draft_token_ids)

        metadata = original_calculate(
            self,
            num_draft_tokens,
            cu_num_scheduled_tokens,
            *args,
            **kwargs,
        )
        if cache_enabled:
            self._vspec_eagle_metadata_cache = (cache_key, metadata)
        return metadata

    model_runner_cls._calc_spec_decode_metadata = calculate
    setattr(model_runner_cls, PATCH_MARKER, True)
    return True
