"""Target active-vocabulary projection and relaxed EAGLE acceptance."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from vllm.v1.outputs import SamplerOutput

PATCH_MARKER = "_vllm_hust_vspec_target_vocab_patched"
CONFIGURED_MARKER = "_vllm_hust_vspec_target_vocab_configured"
logger = logging.getLogger(__name__)


def load_active_vocab_ids(
    path: Path,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        raw_ids = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to load EAGLE active vocabulary: {path}") from exc
    if not isinstance(raw_ids, list) or not raw_ids:
        raise RuntimeError("EAGLE active vocabulary must be a non-empty JSON list")
    if any(type(token_id) is not int for token_id in raw_ids):
        raise RuntimeError("EAGLE active vocabulary IDs must be integers")
    if len(set(raw_ids)) != len(raw_ids):
        raise RuntimeError("EAGLE active vocabulary IDs must be unique")
    if min(raw_ids) < 0 or max(raw_ids) >= vocab_size:
        raise RuntimeError("EAGLE active vocabulary ID is outside the model vocabulary")
    return torch.tensor(raw_ids, dtype=torch.long, device=device)


def _bare_greedy_sampling(sampling_metadata: Any) -> bool:
    if not sampling_metadata.all_greedy:
        return False
    if sampling_metadata.max_num_logprobs is not None:
        return False
    if sampling_metadata.logprob_token_ids:
        return False
    if not sampling_metadata.no_penalties:
        return False
    if sampling_metadata.allowed_token_ids_mask is not None:
        return False
    if sampling_metadata.bad_words_token_ids:
        return False
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        biases = getattr(processor, "biases", None)
        min_tokens = getattr(processor, "min_toks", None)
        if biases or min_tokens or (biases is None and min_tokens is None):
            return False
    thinking_state = sampling_metadata.thinking_budget_state_holder
    return thinking_state is None or not thinking_state.has_tracked_requests()


def _unwrap_model(model: Any) -> Any:
    return model.unwrap() if hasattr(model, "unwrap") else model


def _configure_target_active_vocab(runner: Any) -> None:
    return _configure_target_active_vocab_for_method(
        runner,
        active_ids_environment="VLLM_ASCEND_EAGLE_TARGET_ACTIVE_VOCAB_IDS_PATH",
        required_method="eagle",
        feature_name="EAGLE Target",
        relaxed_topk_environment="VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_TOPK",
        relaxed_prefix_environment=("VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_AFTER_TOKENS"),
        relaxed_margin_environment=("VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_MAX_MARGIN"),
    )


def _configure_target_active_vocab_for_method(
    runner: Any,
    *,
    active_ids_environment: str,
    required_method: str,
    feature_name: str,
    relaxed_topk_environment: str | None = None,
    relaxed_prefix_environment: str | None = None,
    relaxed_margin_environment: str | None = None,
) -> None:
    if getattr(runner, CONFIGURED_MARKER, False):
        return
    active_ids_path = os.environ.get(active_ids_environment)
    runner._eagle_target_active_vocab_ids = None
    runner._eagle_target_active_lm_head_weight = None
    runner._eagle_target_active_lm_head_bias = None
    runner._eagle_relaxed_accept_topk = 1
    runner._eagle_relaxed_accept_after_tokens = 0
    runner._eagle_relaxed_accept_max_margin = None
    runner._eagle_relaxed_draft_mask = None
    if not active_ids_path:
        setattr(runner, CONFIGURED_MARKER, True)
        return
    if runner.speculative_config is None or runner.speculative_config.method != required_method:
        raise RuntimeError(f"{feature_name} active vocabulary requires method={required_method}")
    if runner.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError(f"{feature_name} active vocabulary currently requires TP=1")
    if runner.vllm_config.quant_config is not None:
        raise RuntimeError(f"{feature_name} active vocabulary does not support quantization")

    model = _unwrap_model(runner.model)
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError(f"{feature_name} active vocabulary requires a 2-D LM head")
    active_ids = load_active_vocab_ids(
        Path(active_ids_path),
        int(weight.shape[0]),
        weight.device,
    )
    if active_ids.numel() < 1024:
        raise RuntimeError(f"{feature_name} active vocabulary requires at least 1024 IDs")

    topk = 1
    after_tokens = 0
    max_margin = None
    if relaxed_topk_environment is not None:
        topk = int(os.environ.get(relaxed_topk_environment, "1"))
    if relaxed_prefix_environment is not None:
        after_tokens = int(os.environ.get(relaxed_prefix_environment, "0"))
    if relaxed_margin_environment is not None:
        raw_margin = os.environ.get(relaxed_margin_environment)
        if raw_margin not in {None, ""}:
            max_margin = float(raw_margin)
    if topk <= 0:
        raise ValueError("EAGLE relaxed acceptance top-K must be positive")
    if after_tokens < 0:
        raise ValueError("EAGLE relaxed acceptance prefix must be non-negative")
    if max_margin is not None and (not math.isfinite(max_margin) or max_margin < 0):
        raise ValueError("EAGLE relaxed acceptance max margin must be finite and non-negative")

    active_weight = weight.index_select(0, active_ids).contiguous()
    bias = getattr(lm_head, "bias", None)
    active_bias = (
        bias.index_select(0, active_ids).contiguous() if isinstance(bias, torch.Tensor) else None
    )
    runner._eagle_target_active_vocab_ids = active_ids
    runner._eagle_target_active_lm_head_weight = active_weight
    runner._eagle_target_active_lm_head_bias = active_bias
    runner._eagle_relaxed_accept_topk = topk
    runner._eagle_relaxed_accept_after_tokens = after_tokens
    runner._eagle_relaxed_accept_max_margin = max_margin
    if after_tokens > 0:
        runner._eagle_relaxed_draft_mask = runner._make_buffer(
            runner.max_num_tokens,
            dtype=torch.bool,
        )

    def compute_active_logits(_model: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states, active_weight, active_bias)

    if not hasattr(model, "_vspec_original_compute_logits"):
        model._vspec_original_compute_logits = model.compute_logits
    model.compute_logits = MethodType(compute_active_logits, model)
    setattr(runner, CONFIGURED_MARKER, True)
    logger.info(
        "Enabled %s active vocabulary: active=%d full=%d topk=%d prefix=%d max_margin=%s",
        feature_name,
        active_ids.numel(),
        weight.shape[0],
        topk,
        after_tokens,
        max_margin,
    )


def _sample_target_active_vocab(
    runner: Any,
    logits: torch.Tensor,
    spec_decode_metadata: Any,
) -> SamplerOutput:
    runner.input_batch.update_async_output_token_ids()
    sampling_metadata = runner.input_batch.sampling_metadata
    if not _bare_greedy_sampling(sampling_metadata):
        raise RuntimeError("EAGLE Target active vocabulary requires bare greedy sampling")

    active_ids = runner._eagle_target_active_vocab_ids
    topk = min(runner._eagle_relaxed_accept_topk, logits.shape[-1])
    candidates = None
    candidate_values = None
    if topk == 1:
        target_token_ids = active_ids[logits.argmax(dim=-1)]
    else:
        candidate_values, compact_candidates = torch.topk(
            logits,
            k=topk,
            dim=-1,
        )
        candidates = active_ids[compact_candidates]
        target_token_ids = candidates[:, 0]

    if spec_decode_metadata is None:
        return SamplerOutput(
            sampled_token_ids=target_token_ids.to(torch.int32).view(-1, 1),
            logprobs_tensors=None,
        )

    relaxed_mask = None
    max_margin = runner._eagle_relaxed_accept_max_margin
    if candidate_values is not None and max_margin is not None:
        candidate_margins = candidate_values[:, 0] - candidate_values[:, 1]
        relaxed_mask = candidate_margins[spec_decode_metadata.target_logits_indices] <= max_margin
    after_tokens = runner._eagle_relaxed_accept_after_tokens
    if candidates is not None and after_tokens > 0:
        num_requests = len(spec_decode_metadata.num_draft_tokens)
        request_mask = (
            runner.input_batch.num_tokens_no_spec[:num_requests]
            - runner.input_batch.num_prompt_tokens[:num_requests]
            >= after_tokens
        )
        expanded_mask = np.repeat(
            request_mask,
            spec_decode_metadata.num_draft_tokens,
        )
        mask_buffer = runner._eagle_relaxed_draft_mask
        num_draft_tokens = len(expanded_mask)
        mask_buffer.np[:num_draft_tokens] = expanded_mask
        mask_buffer.copy_to_gpu(num_draft_tokens)
        prefix_mask = mask_buffer.gpu[:num_draft_tokens]
        relaxed_mask = prefix_mask if relaxed_mask is None else relaxed_mask & prefix_mask

    return runner.rejection_sampler.forward_greedy_token_ids(
        spec_decode_metadata,
        target_token_ids,
        sampling_metadata,
        target_token_id_candidates=candidates,
        relaxed_draft_mask=relaxed_mask,
    )


def apply_target_active_vocab_patch(
    *,
    active_ids_environment: str = ("VLLM_ASCEND_EAGLE_TARGET_ACTIVE_VOCAB_IDS_PATH"),
    required_method: str = "eagle",
    feature_name: str = "EAGLE Target",
    relaxed_topk_environment: str | None = ("VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_TOPK"),
    relaxed_prefix_environment: str | None = ("VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_AFTER_TOKENS"),
    relaxed_margin_environment: str | None = ("VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_MAX_MARGIN"),
) -> bool:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, PATCH_MARKER, False):
        return False
    original_load_model = NPUModelRunner.load_model
    original_sample = NPUModelRunner._sample

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        _configure_target_active_vocab_for_method(
            self,
            active_ids_environment=active_ids_environment,
            required_method=required_method,
            feature_name=feature_name,
            relaxed_topk_environment=relaxed_topk_environment,
            relaxed_prefix_environment=relaxed_prefix_environment,
            relaxed_margin_environment=relaxed_margin_environment,
        )
        return result

    def sample(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        active_ids = getattr(self, "_eagle_target_active_vocab_ids", None)
        if active_ids is not None and logits is not None and logits.shape[-1] == active_ids.numel():
            return _sample_target_active_vocab(
                self,
                logits,
                spec_decode_metadata,
            )
        return original_sample(self, logits, spec_decode_metadata)

    def configure_active_vocab(self: Any) -> None:
        _configure_target_active_vocab_for_method(
            self,
            active_ids_environment=active_ids_environment,
            required_method=required_method,
            feature_name=feature_name,
            relaxed_topk_environment=relaxed_topk_environment,
            relaxed_prefix_environment=relaxed_prefix_environment,
            relaxed_margin_environment=relaxed_margin_environment,
        )

    NPUModelRunner.load_model = load_model
    NPUModelRunner._sample = sample
    NPUModelRunner._configure_eagle_target_active_vocab = configure_active_vocab
    setattr(NPUModelRunner, PATCH_MARKER, True)
    return True
