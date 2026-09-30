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


def _load_eagle3_draft_vocab_ids(
    runner: Any,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    drafter = getattr(runner, "drafter", None)
    draft_model = getattr(drafter, "model", None)
    draft_to_target = getattr(draft_model, "draft_id_to_target_id", None)
    if not isinstance(draft_to_target, torch.Tensor) or draft_to_target.ndim != 1:
        raise RuntimeError("EAGLE3 automatic active vocabulary requires a loaded 1-D d2t mapping")
    draft_to_target = draft_to_target.to(device=device, dtype=torch.long)
    active_ids = torch.arange(
        draft_to_target.numel(),
        dtype=torch.long,
        device=device,
    ).add_(draft_to_target)
    if active_ids.unique().numel() != active_ids.numel():
        raise RuntimeError("EAGLE3 d2t mapping contains duplicate target token IDs")
    if active_ids.numel() == 0 or active_ids.min() < 0 or active_ids.max() >= vocab_size:
        raise RuntimeError("EAGLE3 d2t mapping contains an invalid target token ID")
    return active_ids


def _active_vocab_global_argmax(
    logits: torch.Tensor,
    active_ids: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    local_values, local_indices = logits.max(dim=-1)
    local_token_ids = active_ids[local_indices]
    if tp_size == 1:
        return local_token_ids

    from vllm.distributed import tensor_model_parallel_all_gather

    # Target token IDs fit exactly in float32. Packing values and IDs keeps
    # the TP collective independent of the active-vocabulary shard lengths.
    local_pairs = torch.stack(
        (local_values.float(), local_token_ids.float()),
        dim=-1,
    )
    gathered = tensor_model_parallel_all_gather(local_pairs, dim=-1)
    gathered = gathered.view(logits.shape[0], tp_size, 2)
    winning_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(-1, winning_rank).squeeze(-1).to(torch.long)


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
    runner._eagle_target_local_active_vocab_ids = None
    runner._eagle_target_active_lm_head_weight = None
    runner._eagle_target_active_lm_head_bias = None
    runner._eagle_target_full_vocab_size = None
    runner._eagle_target_active_vocab_tp_size = 1
    runner._eagle_relaxed_accept_topk = 1
    runner._eagle_relaxed_accept_after_tokens = 0
    runner._eagle_relaxed_accept_max_margin = None
    runner._eagle_relaxed_draft_mask = None
    if not active_ids_path:
        setattr(runner, CONFIGURED_MARKER, True)
        return
    if runner.speculative_config is None or runner.speculative_config.method != required_method:
        raise RuntimeError(f"{feature_name} active vocabulary requires method={required_method}")
    if runner.vllm_config.quant_config is not None:
        raise RuntimeError(f"{feature_name} active vocabulary does not support quantization")

    model = _unwrap_model(runner.model)
    logits_model = getattr(model, "language_model", model)
    lm_head = getattr(logits_model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError(f"{feature_name} active vocabulary requires a 2-D LM head")
    tp_size = runner.parallel_config.tensor_parallel_size
    shard_indices = getattr(lm_head, "shard_indices", None)
    if tp_size > 1 and shard_indices is None:
        raise RuntimeError(f"{feature_name} TP active vocabulary requires a sharded LM head")
    full_vocab_size = int(getattr(lm_head, "org_vocab_size", weight.shape[0]))
    if active_ids_path == "auto":
        if required_method != "eagle3":
            raise RuntimeError("automatic active vocabulary is only supported for EAGLE3")
        active_ids = _load_eagle3_draft_vocab_ids(
            runner,
            full_vocab_size,
            weight.device,
        )
    else:
        active_ids = load_active_vocab_ids(
            Path(active_ids_path),
            full_vocab_size,
            weight.device,
        )
    if active_ids.numel() < 1024:
        raise RuntimeError(f"{feature_name} active vocabulary requires at least 1024 IDs")

    if tp_size == 1:
        local_active_ids = active_ids
        local_weight_indices = active_ids
    else:
        shard_start = int(shard_indices.org_vocab_start_index)
        shard_end = int(shard_indices.org_vocab_end_index)
        local_mask = (active_ids >= shard_start) & (active_ids < shard_end)
        local_active_ids = active_ids[local_mask]
        if local_active_ids.numel() == 0:
            raise RuntimeError(
                f"{feature_name} active vocabulary has no IDs on TP shard "
                f"[{shard_start}, {shard_end})"
            )
        local_weight_indices = local_active_ids - shard_start

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
    if tp_size > 1 and topk > 1:
        raise RuntimeError(f"{feature_name} relaxed top-K is not supported with TP>1")

    active_weight = weight.index_select(0, local_weight_indices).contiguous()
    bias = getattr(lm_head, "bias", None)
    active_bias = (
        bias.index_select(0, local_weight_indices).contiguous()
        if isinstance(bias, torch.Tensor)
        else None
    )
    runner._eagle_target_active_vocab_ids = active_ids
    runner._eagle_target_local_active_vocab_ids = local_active_ids
    runner._eagle_target_active_lm_head_weight = active_weight
    runner._eagle_target_active_lm_head_bias = active_bias
    runner._eagle_target_full_vocab_size = full_vocab_size
    runner._eagle_target_active_vocab_tp_size = tp_size
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
        local_active_ids.numel(),
        full_vocab_size,
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
    active_ids = runner._eagle_target_local_active_vocab_ids
    full_vocab_size = runner._eagle_target_full_vocab_size
    tp_size = runner._eagle_target_active_vocab_tp_size
    if not _bare_greedy_sampling(sampling_metadata):
        if tp_size > 1:
            raise RuntimeError(
                "TP active vocabulary requires bare greedy sampling without penalties"
            )
        from .eagle_rejection import _has_active_non_argmax_processor

        repetition_only = (
            sampling_metadata.all_greedy
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logprob_token_ids
            and getattr(sampling_metadata, "_vspec_repetition_only", False)
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
            and not _has_active_non_argmax_processor(sampling_metadata)
        )
        thinking_state = sampling_metadata.thinking_budget_state_holder
        repetition_only &= thinking_state is None or not thinking_state.has_tracked_requests()
        if not repetition_only:
            raise RuntimeError(
                "Target active vocabulary requires greedy sampling with no processor "
                "other than repetition penalty"
            )

        if spec_decode_metadata is None:
            from .draft_repetition import fused_draft_repetition_greedy
            from .eagle_draft import ActiveVocabLogits

            runner._vspec_draft_full_vocab_size = full_vocab_size
            target_token_ids = fused_draft_repetition_greedy(
                runner,
                ActiveVocabLogits(logits, active_ids),
                sampling_metadata,
                [],
            )
            if target_token_ids is None:
                raise RuntimeError(
                    "Target active vocabulary could not apply exact repetition penalty"
                )
            return SamplerOutput(
                sampled_token_ids=target_token_ids.to(torch.int32).view(-1, 1),
                logprobs_tensors=None,
            )

        from .draft_repetition import fused_repetition_greedy

        fused_tokens = fused_repetition_greedy(
            runner.rejection_sampler,
            spec_decode_metadata,
            logits,
            sampling_metadata,
            active_ids=active_ids,
            full_vocab_size=full_vocab_size,
        )
        if fused_tokens is None:
            raise RuntimeError("Target active vocabulary could not apply exact repetition penalty")
        target_token_ids, bonus_token_ids, _ = fused_tokens
        return runner.rejection_sampler.forward_greedy_token_ids(
            spec_decode_metadata,
            target_token_ids,
            sampling_metadata,
            target_rows_selected=True,
            bonus_token_ids=bonus_token_ids,
        )

    topk = min(runner._eagle_relaxed_accept_topk, logits.shape[-1])
    candidates = None
    candidate_values = None
    if topk == 1:
        target_token_ids = _active_vocab_global_argmax(logits, active_ids, tp_size)
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
        active_ids = getattr(self, "_eagle_target_local_active_vocab_ids", None)
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
