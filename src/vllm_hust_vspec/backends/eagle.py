"""Compatibility and correctness patches for EAGLE and EAGLE3."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from ..config import PluginSettings

PATCH_MARKER = "_vllm_hust_vspec_eagle_patched"

HOST_ENV_FEATURES = {
    "VLLM_ASCEND_EAGLE_DRAFT_IO_TRACE_DIR": (
        "vllm_ascend.spec_decode.llm_base_proposer",
        "EAGLE Draft internal IO trace",
    ),
    "VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL": (
        "vllm_ascend.worker.model_runner_v1",
        "EAGLE uniform-state kernel",
    ),
    "VLLM_ASCEND_EAGLE_TARGET_WIDTH": (
        "vllm_ascend.worker.model_runner_v1",
        "EAGLE Target-width verification",
    ),
    "VLLM_ASCEND_EAGLE_TREE_GRAPH_COMMIT": (
        "vllm_ascend.worker.model_runner_v1",
        "EAGLE tree Graph commit",
    ),
}


def _require_attribute(owner: Any, name: str, feature: str) -> None:
    if not hasattr(owner, name):
        raise RuntimeError(
            f"vSpec {feature} requires {owner.__name__}.{name}; "
            "use the matching vllm-hust and vllm-ascend-hust revisions"
        )


def _validate_host_environment_features() -> None:
    import importlib

    for environment_name, (module_name, feature) in HOST_ENV_FEATURES.items():
        raw_value = os.environ.get(environment_name)
        if raw_value in {None, "", "0", "false", "False"}:
            continue
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__ or "")
        try:
            source = module_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"vSpec cannot validate host support for {feature}") from exc
        if environment_name not in source:
            raise RuntimeError(
                f"vSpec {feature} requires host support for "
                f"{environment_name}; use the matching vllm-ascend-hust revision"
            )


def _validate_runtime(settings: PluginSettings) -> None:
    from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

    _validate_host_environment_features()
    _require_attribute(AscendEagleProposer, "propose", settings.method)
    if settings.eagle_tree_width > 1:
        _require_attribute(AscendEagleProposer, "build_tree", "EAGLE tree")
        _require_attribute(AscendRejectionSampler, "_forward_tree", "EAGLE tree")


def apply_eagle_patches(settings: PluginSettings) -> bool:
    """Apply the Qwen2 EAGLE correctness patch and validate optimized APIs."""
    _validate_runtime(settings)
    from .eagle_tp import apply_replicated_draft_tp_patch

    applied = apply_replicated_draft_tp_patch()
    if os.environ.get("VSPEC_EAGLE_TENSOR_SEQ_LENS") == "1":
        from .eagle_tensor_fia import apply_tensor_seq_lens_patch

        applied = apply_tensor_seq_lens_patch() or applied
    parallel_update_workers = int(os.environ.get("VSPEC_EAGLE_PARALLEL_GRAPH_UPDATES", "0"))
    if parallel_update_workers > 1:
        from .eagle_parallel_update import apply_parallel_graph_update_patch

        applied = apply_parallel_graph_update_patch(parallel_update_workers) or applied
    if os.environ.get("VSPEC_EAGLE_DRAFT_BODY_W8A16") == "1":
        from .eagle_body_quant import apply_eagle_body_quantization_patch

        applied = apply_eagle_body_quantization_patch() or applied
    target_w8a16 = os.environ.get("VSPEC_TARGET_BODY_W8A16") == "1"
    target_w8a8 = os.environ.get("VSPEC_TARGET_BODY_W8A8") == "1"
    if target_w8a16 and target_w8a8:
        raise RuntimeError("Target body W8A16 and W8A8 are mutually exclusive")
    if target_w8a16 or target_w8a8:
        from .target_body_quant import apply_target_body_quantization_patch

        applied = (
            apply_target_body_quantization_patch("w8a8" if target_w8a8 else "w8a16") or applied
        )
    if settings.eagle_tree_width == 1:
        from .eagle_rejection import apply_linear_rejection_patch

        applied = apply_linear_rejection_patch()
    if os.environ.get("VLLM_ASCEND_GRAPH_EVENT_ORDERING") == "1":
        from .eagle_graph import apply_graph_event_ordering_patch

        applied = apply_graph_event_ordering_patch() or applied
    if os.environ.get("VLLM_ASCEND_EAGLE_DISABLE_DRAFT_TORCH_COMPILE") == "1":
        from .eagle_host import apply_draft_compile_control_patch

        applied = apply_draft_compile_control_patch() or applied
    if os.environ.get("VLLM_ASCEND_EAGLE_ISOLATE_SHARED_MODULES") == "1":
        from .eagle_host import apply_shared_module_isolation_patch

        applied = apply_shared_module_isolation_patch() or applied
    if os.environ.get("VLLM_ASCEND_EAGLE_PRESERVE_TARGET_HIDDEN") == "1":
        from .eagle_host import apply_preserve_target_hidden_patch

        applied = apply_preserve_target_hidden_patch() or applied
    if os.environ.get("VLLM_ASCEND_EAGLE_SPEC_METADATA_CACHE") == "1":
        from .eagle_metadata import apply_metadata_cache_patch

        applied = apply_metadata_cache_patch() or applied
    if settings.eagle_target_active_vocab:
        from .eagle_target import apply_target_active_vocab_patch

        applied = apply_target_active_vocab_patch() or applied
    if settings.eagle_draft_active_vocab:
        from .eagle_draft import apply_draft_active_vocab_patch

        applied = apply_draft_active_vocab_patch() or applied
    runtime_patch_requested = any(
        os.environ.get(name)
        for name in (
            "VLLM_ASCEND_EAGLE_ZERO_DRAFT_KV_FIRST_STEP",
            "VLLM_ASCEND_EAGLE_DRAFT_TRACE",
            "VLLM_ASCEND_EAGLE_TARGET_HIDDEN_TRACE_DIR",
            "VLLM_ASCEND_EAGLE_TARGET_ARGMAX_TRACE_PATH",
        )
    )
    if runtime_patch_requested:
        from .eagle_runtime import apply_eagle_runtime_patch

        applied = apply_eagle_runtime_patch() or applied
    if settings.method == "eagle3":
        return applied

    try:
        from vllm.model_executor.models.qwen2_eagle import Qwen2Model
    except ImportError:
        from ..models.qwen2_eagle import Qwen2Model

    if getattr(Qwen2Model, PATCH_MARKER, False):
        return applied

    original_load_weights = Qwen2Model.load_weights

    def load_weights(
        self: Any,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # Some Qwen2 EAGLE checkpoints omit these optional biases. Graph
        # replay must not observe allocator contents for missing parameters.
        with torch.no_grad():
            for layer in self.layers:
                bias = layer.self_attn.qkv_proj.bias
                if bias is not None:
                    bias.zero_()
        return original_load_weights(self, weights)

    Qwen2Model.load_weights = load_weights
    setattr(Qwen2Model, PATCH_MARKER, True)
    return True
