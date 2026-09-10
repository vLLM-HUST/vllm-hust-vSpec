"""EAGLE runtime diagnostics and first-step KV initialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

PATCH_MARKER = "_vllm_hust_vspec_runtime_patched"


def _zero_draft_kv_once(proposer: Any) -> None:
    if getattr(proposer, "_vspec_draft_kv_zeroed", False):
        return
    context = proposer.vllm_config.compilation_config.static_forward_context
    for layer_name in proposer.attn_layer_names:
        kv_cache = context[layer_name].kv_cache
        if isinstance(kv_cache, torch.Tensor):
            kv_cache.zero_()
        else:
            for cache_tensor in kv_cache:
                cache_tensor.zero_()
    proposer._vspec_draft_kv_zeroed = True


def _trace_target_inputs(proposer: Any, kwargs: dict[str, Any]) -> None:
    trace_dir = os.environ.get("VLLM_ASCEND_EAGLE_TARGET_HIDDEN_TRACE_DIR")
    trace_index = getattr(proposer, "_vspec_target_trace_index", 0)
    if not trace_dir or trace_index >= 8:
        return
    hidden_states = kwargs.get("target_hidden_states")
    token_ids = kwargs.get("target_token_ids")
    positions = kwargs.get("target_positions")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            hidden_states,
            token_ids,
            positions,
        )
    ):
        return
    torch.npu.current_stream().synchronize()
    trace_rows = min(int(hidden_states.shape[0]), 64)
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_hidden_states": hidden_states[:trace_rows].float().cpu(),
            "target_token_ids": token_ids[:trace_rows].cpu(),
            "target_positions": positions[..., :trace_rows].cpu(),
        },
        trace_path / f"step_{trace_index:02d}.pt",
    )
    proposer._vspec_target_trace_index = trace_index + 1


def apply_eagle_runtime_patch() -> bool:
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, False):
        return False
    original_propose = AscendSpecDecodeBaseProposer._propose
    original_sample = NPUModelRunner._sample

    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        if os.environ.get("VLLM_ASCEND_EAGLE_ZERO_DRAFT_KV_FIRST_STEP") == "1":
            _zero_draft_kv_once(self)
        _trace_target_inputs(self, kwargs)
        draft_token_ids = original_propose(self, *args, **kwargs)
        trace_index = getattr(self, "_vspec_draft_trace_index", 0)
        if os.environ.get("VLLM_ASCEND_EAGLE_DRAFT_TRACE") == "1" and trace_index < 8:
            print(
                "[VLLM_ASCEND_EAGLE_DRAFT_TRACE] "
                f"step={trace_index} shape={tuple(draft_token_ids.shape)} "
                f"rows={draft_token_ids[:8].cpu().tolist()}",
                flush=True,
            )
            self._vspec_draft_trace_index = trace_index + 1
        return draft_token_ids

    def sample(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        trace_path = os.environ.get("VLLM_ASCEND_EAGLE_TARGET_ARGMAX_TRACE_PATH")
        if (
            trace_path
            and logits is not None
            and getattr(self, "_eagle_target_active_vocab_ids", None) is None
        ):
            target_ids = logits.argmax(dim=-1).cpu().tolist()
            with Path(trace_path).open("a", encoding="utf-8") as trace_file:
                trace_file.write(",".join(str(int(token_id)) for token_id in target_ids) + "\n")
        return original_sample(self, logits, spec_decode_metadata)

    AscendSpecDecodeBaseProposer._propose = propose
    NPUModelRunner._sample = sample
    setattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, True)
    return True
