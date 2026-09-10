"""vLLM general plugin entry point for vSpec."""

from __future__ import annotations

import logging

from ._version import __version__
from .config import PluginSettings

__all__ = ["PluginSettings", "__version__", "register"]

logger = logging.getLogger(__name__)


def register() -> None:
    """Apply the plugin in every vLLM process when explicitly activated."""
    settings = PluginSettings.from_environment()
    if not settings.enabled:
        return

    from .models import register_models

    registered_models = register_models()

    from .patches import apply_patches

    applied = apply_patches(settings)
    logger.info(
        "Enabled vllm-hust-vSpec: method=%s, local_patch=%s, "
        "merged_full=%s, shared_tokenizer_padding=%s, max_num_seqs=%d, "
        "adaptive=%s, registered_models=%s",
        settings.method,
        applied,
        settings.use_merged_full,
        settings.assume_shared_tokenizer,
        settings.max_num_seqs,
        settings.adaptive_speculation,
        registered_models,
    )
