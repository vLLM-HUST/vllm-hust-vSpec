"""Linear EAGLE rejection path without per-step device-to-host statistics."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from functools import lru_cache
from typing import Any

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from ..config import (
    ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS,
    ENV_CONFIDENCE_ACCEPT_FROM_POSITION,
    ENV_CONFIDENCE_ACCEPT_MARGIN,
    ENV_CONFIDENCE_PROTECTED_TOKEN_IDS,
)

PATCH_MARKER = "_vllm_hust_vspec_linear_rejection_patched"
SAMPLING_METADATA_PATCH_MARKER = "_vllm_hust_vspec_repetition_metadata_patched"
EXACT_REPETITION_TOPK_ENV = "HUST_VSPEC_DRAFT_EXACT_REPETITION_TOPK"
LEGACY_EXACT_REPETITION_TOPK_ENV = "VSPEC_DRAFT_EXACT_REPETITION_TOPK"
EXACT_REPETITION_SYNC_PROOF_ENV = (
    "HUST_VSPEC_DRAFT_EXACT_REPETITION_SYNC_PROOF"
)
SPARSE_REPETITION_TOPK_ENV = "VSPEC_DRAFT_SPARSE_REPETITION_TOPK"
FUSED_REPETITION_ENV = "VSPEC_DRAFT_FUSED_REPETITION"
NATIVE_REPETITION_ENV = "VSPEC_DRAFT_NATIVE_REPETITION"
CONFIDENCE_ACCEPT_MARGIN_ENV = ENV_CONFIDENCE_ACCEPT_MARGIN
CONFIDENCE_ACCEPT_FROM_POSITION_ENV = ENV_CONFIDENCE_ACCEPT_FROM_POSITION
CONFIDENCE_ACCEPT_AFTER_TOKENS_ENV = ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS
CONFIDENCE_PROTECTED_TOKEN_IDS_ENV = ENV_CONFIDENCE_PROTECTED_TOKEN_IDS
_LEGACY_CONFIDENCE_ENVIRONMENTS = {
    CONFIDENCE_ACCEPT_MARGIN_ENV: "VSPEC_EAGLE_CONFIDENCE_ACCEPT_MARGIN",
    CONFIDENCE_ACCEPT_FROM_POSITION_ENV: "VSPEC_EAGLE_CONFIDENCE_ACCEPT_FROM_POSITION",
    CONFIDENCE_ACCEPT_AFTER_TOKENS_ENV: "VSPEC_EAGLE_CONFIDENCE_ACCEPT_AFTER_TOKENS",
    CONFIDENCE_PROTECTED_TOKEN_IDS_ENV: "VSPEC_EAGLE_CONFIDENCE_PROTECTED_TOKEN_IDS",
}
CONFIDENCE_ACCEPT_ENABLED_ATTR = "_vspec_eagle_confidence_accept_enabled"
_POSITION_MASK_CACHE: dict[tuple[str, tuple[int, ...], int], torch.Tensor] = {}
_REQUEST_MASK_CACHE: dict[tuple[str, tuple[int, ...], tuple[bool, ...]], torch.Tensor] = {}


def _confidence_environment(name: str, default: str | None = None) -> str | None:
    return os.environ.get(
        name,
        os.environ.get(_LEGACY_CONFIDENCE_ENVIRONMENTS[name], default),
    )


def _confidence_position_mask(
    metadata: Any,
    device: torch.device,
    min_position: int,
) -> torch.Tensor:
    lengths = tuple(int(length) for length in metadata.num_draft_tokens)
    key = (str(device), lengths, min_position)
    cached = _POSITION_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    positions = [position >= min_position for length in lengths for position in range(length)]
    mask = torch.tensor(positions, dtype=torch.bool, device=device)
    if len(_POSITION_MASK_CACHE) >= 128:
        _POSITION_MASK_CACHE.clear()
    _POSITION_MASK_CACHE[key] = mask
    return mask


def _filter_confidence_accept_mask(
    relaxed_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    metadata: Any,
    min_position: int,
    protected_token_ids: tuple[int, ...],
    min_generated_tokens: int,
    output_token_ids: list[list[int]] | None,
) -> torch.Tensor:
    if min_position:
        relaxed_mask &= _confidence_position_mask(
            metadata,
            relaxed_mask.device,
            min_position,
        )
    if min_generated_tokens:
        if output_token_ids is None:
            raise ValueError("output token IDs are required for confidence prefix protection")
        lengths = tuple(int(length) for length in metadata.num_draft_tokens)
        request_enabled = tuple(
            index < len(output_token_ids)
            and len(output_token_ids[index]) >= min_generated_tokens
            for index in range(len(lengths))
        )
        key = (str(relaxed_mask.device), lengths, request_enabled)
        request_mask = _REQUEST_MASK_CACHE.get(key)
        if request_mask is None:
            request_mask = torch.tensor(
                [
                    enabled
                    for enabled, length in zip(
                        request_enabled,
                        lengths,
                        strict=True,
                    )
                    for _ in range(length)
                ],
                dtype=torch.bool,
                device=relaxed_mask.device,
            )
            if len(_REQUEST_MASK_CACHE) >= 128:
                _REQUEST_MASK_CACHE.clear()
            _REQUEST_MASK_CACHE[key] = request_mask
        relaxed_mask &= request_mask
    for token_id in protected_token_ids:
        relaxed_mask &= (target_token_ids != token_id) & (draft_token_ids != token_id)
    return relaxed_mask


def _confidence_accept_inputs(
    logits: torch.Tensor,
    metadata: Any,
    max_margin: float,
    min_position: int = 0,
    protected_token_ids: tuple[int, ...] = (),
    min_generated_tokens: int = 0,
    output_token_ids: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_values, target_token_ids = logits.max(dim=-1)
    target_indices = metadata.target_logits_indices
    draft_token_ids = metadata.draft_token_ids.long()
    draft_values = logits[target_indices, draft_token_ids]
    relaxed_mask = target_values[target_indices] - draft_values <= max_margin
    target_argmax = target_token_ids[target_indices]
    return target_token_ids, _filter_confidence_accept_mask(
        relaxed_mask,
        target_argmax,
        draft_token_ids,
        metadata,
        min_position,
        protected_token_ids,
        min_generated_tokens,
        output_token_ids,
    )


def _confidence_accept_selected_inputs(
    target_logits: torch.Tensor,
    metadata: Any,
    max_margin: float,
    min_position: int = 0,
    protected_token_ids: tuple[int, ...] = (),
    min_generated_tokens: int = 0,
    output_token_ids: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate relaxed acceptance after target rows have been selected."""
    target_values, target_token_ids = target_logits.max(dim=-1)
    draft_token_ids = metadata.draft_token_ids.long()
    row_indices = torch.arange(
        draft_token_ids.shape[0],
        device=target_logits.device,
    )
    draft_values = target_logits[row_indices, draft_token_ids]
    relaxed_mask = target_values - draft_values <= max_margin
    return target_token_ids, _filter_confidence_accept_mask(
        relaxed_mask,
        target_token_ids,
        draft_token_ids,
        metadata,
        min_position,
        protected_token_ids,
        min_generated_tokens,
        output_token_ids,
    )


def _confidence_accept_margin() -> float | None:
    raw_margin = _confidence_environment(CONFIDENCE_ACCEPT_MARGIN_ENV)
    if raw_margin in {None, ""}:
        return None
    max_margin = float(raw_margin)
    if not math.isfinite(max_margin) or max_margin < 0:
        raise ValueError(f"{CONFIDENCE_ACCEPT_MARGIN_ENV} must be finite and non-negative")
    return max_margin


def _confidence_accept_from_position() -> int:
    raw_position = _confidence_environment(
        CONFIDENCE_ACCEPT_FROM_POSITION_ENV,
        "0",
    )
    min_position = int(raw_position)
    if min_position < 0:
        raise ValueError(f"{CONFIDENCE_ACCEPT_FROM_POSITION_ENV} must be non-negative")
    return min_position


def _confidence_accept_after_tokens() -> int:
    raw_tokens = _confidence_environment(CONFIDENCE_ACCEPT_AFTER_TOKENS_ENV, "0")
    min_generated_tokens = int(raw_tokens)
    if min_generated_tokens < 0:
        raise ValueError(f"{CONFIDENCE_ACCEPT_AFTER_TOKENS_ENV} must be non-negative")
    return min_generated_tokens


@lru_cache(maxsize=1)
def _confidence_protected_token_ids() -> tuple[int, ...]:
    raw_ids = _confidence_environment(CONFIDENCE_PROTECTED_TOKEN_IDS_ENV, "")
    if not raw_ids.strip():
        return ()
    token_ids = tuple(int(value.strip()) for value in raw_ids.split(","))
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError(
            f"{CONFIDENCE_PROTECTED_TOKEN_IDS_ENV} must contain non-negative token IDs"
        )
    return token_ids


def _forward_linear_eagle(
    self: Any,
    metadata: Any,
    draft_probs: torch.Tensor | None,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> SamplerOutput:
    """Run the native Ascend verifier without nano-PEARL host statistics."""
    from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN
    from vllm_ascend.sample import rejection_sampler as ascend_rejection

    if metadata.max_spec_len > MAX_SPEC_LEN:
        raise ValueError(f"speculative length {metadata.max_spec_len} exceeds {MAX_SPEC_LEN}")

    _trace_exact_topk_eligibility(self, sampling_metadata)

    from .eagle_target import _bare_greedy_sampling

    if _bare_greedy_sampling(sampling_metadata):
        max_margin = (
            _confidence_accept_margin()
            if getattr(self, CONFIDENCE_ACCEPT_ENABLED_ATTR, True)
            else None
        )
        if max_margin is not None:
            target_ids, relaxed_mask = _confidence_accept_inputs(
                logits,
                metadata,
                max_margin,
                _confidence_accept_from_position(),
                _confidence_protected_token_ids(),
                _confidence_accept_after_tokens(),
                sampling_metadata.output_token_ids,
            )
            return _forward_greedy_token_ids(
                self,
                metadata,
                target_ids,
                sampling_metadata,
                relaxed_draft_mask=relaxed_mask,
            )
        # The generic verifier converts the complete target matrix to FP32 and
        # clones it before discovering that greedy sampling only needs token
        # IDs. Keep the strict argmax semantics while avoiding those two large
        # vocabulary tensors.
        return _forward_greedy_token_ids(
            self,
            metadata,
            logits.argmax(dim=-1),
            sampling_metadata,
        )

    # Greedy requests with penalties still only need processed token IDs. The
    # generic path retains complete FP32 logits for nonexistent logprobs and
    # sends them through the probability-oriented rejection interface.
    if (
        sampling_metadata.all_greedy
        and sampling_metadata.max_num_logprobs is None
        and not sampling_metadata.logprob_token_ids
    ):
        native_tokens = _native_repetition_greedy(
            self,
            metadata,
            logits,
            sampling_metadata,
        )
        if native_tokens is not None:
            target_token_ids, bonus_token_ids = native_tokens
            return _forward_greedy_token_ids(
                self,
                metadata,
                target_token_ids,
                sampling_metadata,
                target_rows_selected=True,
                bonus_token_ids=bonus_token_ids,
            )

        exact_topk_tokens = _exact_topk_repetition_greedy(
            self,
            metadata,
            logits,
            sampling_metadata,
        )
        if exact_topk_tokens is not None:
            target_token_ids, bonus_token_ids = exact_topk_tokens
            return _forward_greedy_token_ids(
                self,
                metadata,
                target_token_ids,
                sampling_metadata,
                target_rows_selected=True,
                bonus_token_ids=bonus_token_ids,
            )

        fused_tokens = _fused_repetition_greedy(
            self,
            metadata,
            logits,
            sampling_metadata,
        )
        if fused_tokens is not None:
            target_token_ids, bonus_token_ids, relaxed_mask = fused_tokens
            return _forward_greedy_token_ids(
                self,
                metadata,
                target_token_ids,
                sampling_metadata,
                target_rows_selected=True,
                bonus_token_ids=bonus_token_ids,
                relaxed_draft_mask=relaxed_mask,
            )

        sparse_tokens = _sparse_repetition_greedy(
            metadata,
            logits,
            sampling_metadata,
        )
        if sparse_tokens is not None:
            target_token_ids, bonus_token_ids = sparse_tokens
            return _forward_greedy_token_ids(
                self,
                metadata,
                target_token_ids,
                sampling_metadata,
                target_rows_selected=True,
                bonus_token_ids=bonus_token_ids,
            )

        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(sampling_metadata, max_num_logprobs=None),
            predict_bonus_token=True,
        )

        target_logits = logits[metadata.target_logits_indices].to(torch.float32)
        target_logits = self.apply_logits_processors(
            target_logits,
            sampling_metadata,
            metadata,
        )
        max_margin = (
            _confidence_accept_margin()
            if getattr(self, CONFIDENCE_ACCEPT_ENABLED_ATTR, True)
            else None
        )
        relaxed_mask = None
        if max_margin is None:
            target_token_ids = target_logits.argmax(dim=-1)
        else:
            target_token_ids, relaxed_mask = _confidence_accept_selected_inputs(
                target_logits,
                metadata,
                max_margin,
                _confidence_accept_from_position(),
                _confidence_protected_token_ids(),
                _confidence_accept_after_tokens(),
                sampling_metadata.output_token_ids,
            )
        return _forward_greedy_token_ids(
            self,
            metadata,
            target_token_ids,
            sampling_metadata,
            target_rows_selected=True,
            bonus_token_ids=bonus_sampler_output.sampled_token_ids,
            relaxed_draft_mask=relaxed_mask,
        )

    bonus_logits = logits[metadata.bonus_logits_indices]
    bonus_sampler_output = self.sampler(
        logits=bonus_logits,
        sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
        predict_bonus_token=True,
        logprobs_mode_override=(
            "processed_logits" if self.is_processed_logprobs_mode else "raw_logits"
        ),
    )
    bonus_token_ids = bonus_sampler_output.sampled_token_ids

    raw_target_logits = logits[metadata.target_logits_indices].to(torch.float32)
    target_logits = raw_target_logits
    if not self.is_processed_logprobs_mode:
        target_logits = target_logits.clone()
    target_logits = self.apply_logits_processors(
        target_logits,
        sampling_metadata,
        metadata,
    )
    target_logits = ascend_rejection.apply_sampling_constraints(
        target_logits,
        metadata.cu_num_draft_tokens,
        sampling_metadata,
        self.top_k,
    )

    output_token_ids = ascend_rejection.rejection_sample(
        metadata.draft_token_ids,
        metadata.num_draft_tokens,
        metadata.max_spec_len,
        metadata.cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        ori_target_logits=raw_target_logits,
    )

    # EAGLE does not consume the custom nano-PEARL verification payload. The
    # old payload builder called .item() for every row and synchronized the NPU.
    self.last_verify_results = None
    logprobs_tensors = None
    if sampling_metadata.max_num_logprobs is not None:
        logprobs_tensors = self._get_logprobs_tensors(
            sampling_metadata.max_num_logprobs,
            metadata,
            logits,
            (target_logits if self.is_processed_logprobs_mode else raw_target_logits),
            bonus_sampler_output.logprobs_tensors.logprobs,
            output_token_ids,
        )

    return SamplerOutput(
        sampled_token_ids=output_token_ids,
        logprobs_tensors=logprobs_tensors,
    )


def _exact_topk_repetition_greedy(
    owner: Any,
    metadata: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if os.environ.get("HUST_VSPEC_METHOD") != "draft_model":
        return None
    try:
        candidate_count = int(
            os.environ.get(
                EXACT_REPETITION_TOPK_ENV,
                os.environ.get(LEGACY_EXACT_REPETITION_TOPK_ENV, "0"),
            )
        )
    except ValueError:
        return None
    if candidate_count <= 0 or not getattr(
        sampling_metadata,
        "_vspec_repetition_only",
        False,
    ):
        return None
    if _confidence_accept_margin() is not None:
        return None
    if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
        return None
    if getattr(sampling_metadata, "bad_words_token_ids", None):
        return None
    if _has_active_non_argmax_processor(sampling_metadata):
        return None
    holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
    if holder is not None and holder.has_tracked_requests():
        return None

    from .draft_repetition import exact_topk_repetition_greedy

    return exact_topk_repetition_greedy(
        owner,
        metadata,
        logits,
        sampling_metadata,
        candidate_count,
        synchronize_proof=(
            os.environ.get(EXACT_REPETITION_SYNC_PROOF_ENV, "1") == "1"
        ),
    )


def _native_repetition_greedy(
    owner: Any,
    metadata: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if os.environ.get("HUST_VSPEC_METHOD") != "draft_model":
        return None
    if os.environ.get(NATIVE_REPETITION_ENV) != "1":
        return None
    if not getattr(sampling_metadata, "_vspec_repetition_only", False):
        return None
    if _confidence_accept_margin() is not None:
        return None
    if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
        return None
    if getattr(sampling_metadata, "bad_words_token_ids", None):
        return None
    if _has_active_non_argmax_processor(sampling_metadata):
        return None
    holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
    if holder is not None and holder.has_tracked_requests():
        return None

    from .draft_repetition import native_sparse_repetition_greedy

    return native_sparse_repetition_greedy(
        owner,
        metadata,
        logits,
        sampling_metadata,
    )


def _fused_repetition_greedy(
    owner: Any,
    metadata: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    if os.environ.get("HUST_VSPEC_METHOD") != "draft_model":
        return None
    if os.environ.get(FUSED_REPETITION_ENV) != "1":
        return None
    if not getattr(sampling_metadata, "_vspec_repetition_only", False):
        return None
    if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
        return None
    if getattr(sampling_metadata, "bad_words_token_ids", None):
        return None
    if _has_active_non_argmax_processor(sampling_metadata):
        return None
    holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
    if holder is not None and holder.has_tracked_requests():
        return None

    from .draft_repetition import fused_repetition_greedy

    max_margin = _confidence_accept_margin()
    result = fused_repetition_greedy(
        owner,
        metadata,
        logits,
        sampling_metadata,
        max_margin=max_margin,
    )
    if result is None:
        return None
    target_token_ids, bonus_token_ids, relaxed_mask = result
    if relaxed_mask is not None:
        relaxed_mask = _filter_confidence_accept_mask(
            relaxed_mask,
            target_token_ids,
            metadata.draft_token_ids.long(),
            metadata,
            _confidence_accept_from_position(),
            _confidence_protected_token_ids(),
            _confidence_accept_after_tokens(),
            sampling_metadata.output_token_ids,
        )
    return target_token_ids, bonus_token_ids, relaxed_mask


def _sparse_repetition_greedy(
    metadata: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select greedy tokens without materializing full-vocabulary penalty masks."""
    if os.environ.get("HUST_VSPEC_METHOD") != "draft_model":
        return None
    try:
        candidate_count = int(os.environ.get(SPARSE_REPETITION_TOPK_ENV, "0"))
    except ValueError:
        return None
    if candidate_count <= 0 or not getattr(
        sampling_metadata,
        "_vspec_repetition_only",
        False,
    ):
        return None
    if _confidence_accept_margin() is not None:
        return None
    if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
        return None
    if getattr(sampling_metadata, "bad_words_token_ids", None):
        return None
    if _has_active_non_argmax_processor(sampling_metadata):
        return None
    holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
    if holder is not None and holder.has_tracked_requests():
        return None
    if sampling_metadata.prompt_token_ids is None:
        return None
    spec_token_ids = getattr(sampling_metadata, "spec_token_ids", None)
    if spec_token_ids is None or not hasattr(metadata, "cu_num_draft_tokens"):
        return None

    from vllm.utils.platform_utils import is_pin_memory_available
    from vllm.utils.torch_utils import make_tensor_with_pad
    from vllm_ascend.sample import rejection_sampler as ascend_rejection

    num_requests = len(metadata.num_draft_tokens)
    target_rows = int(metadata.target_logits_indices.numel())
    target_histories = []
    for output, spec in zip(
        sampling_metadata.output_token_ids,
        spec_token_ids,
        strict=True,
    ):
        if not spec:
            continue
        target_histories.append(output)
        for position in range(len(spec) - 1):
            target_histories.append([*output, *spec[: position + 1]])
    bonus_histories = [
        [*output, *spec] if spec else output
        for output, spec in zip(
            sampling_metadata.output_token_ids,
            spec_token_ids,
            strict=True,
        )
    ]
    if len(target_histories) != target_rows or len(bonus_histories) != num_requests:
        return None

    request_indices = torch.arange(
        num_requests,
        device=logits.device,
        dtype=torch.long,
    )
    target_repeat_indices = ascend_rejection.expand_batch_to_tokens(
        request_indices,
        metadata.cu_num_draft_tokens,
        target_rows,
    )
    repeat_indices = torch.cat((target_repeat_indices, request_indices))
    row_indices = torch.cat(
        (metadata.target_logits_indices, metadata.bonus_logits_indices)
    )
    candidate_count = min(candidate_count, logits.shape[-1])
    candidate_values, candidate_ids = logits[row_indices].topk(
        candidate_count,
        dim=-1,
    )
    candidate_values = candidate_values.to(torch.float32)

    prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
    prompt_seen = (
        candidate_ids.unsqueeze(-1) == prompt_token_ids.unsqueeze(1)
    ).any(dim=-1)
    histories = [*target_histories, *bonus_histories]
    output_token_ids = make_tensor_with_pad(
        histories,
        pad=logits.shape[-1],
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    ).to(logits.device, non_blocking=True)
    output_seen = (
        candidate_ids.unsqueeze(-1) == output_token_ids.unsqueeze(1)
    ).any(dim=-1)
    seen = prompt_seen | output_seen
    penalties = sampling_metadata.repetition_penalties[repeat_indices].unsqueeze(-1)
    penalized_values = torch.where(
        candidate_values > 0,
        candidate_values / penalties,
        candidate_values * penalties,
    )
    candidate_values = torch.where(seen, penalized_values, candidate_values)
    winner_positions = candidate_values.argmax(dim=-1, keepdim=True)
    token_ids = candidate_ids.gather(-1, winner_positions).squeeze(-1)
    return (
        token_ids[:target_rows],
        token_ids[target_rows:].to(torch.int32).unsqueeze(-1),
    )


def _has_active_non_argmax_processor(sampling_metadata: Any) -> bool:
    logits_processors = getattr(sampling_metadata, "logitsprocs", None)
    if logits_processors is None:
        return True
    for processor in logits_processors.non_argmax_invariant:
        biases = getattr(processor, "biases", None)
        min_tokens = getattr(processor, "min_toks", None)
        if biases or min_tokens or (biases is None and min_tokens is None):
            return True
    return False


def _forward_greedy_token_ids(
    self: Any,
    metadata: Any,
    target_token_ids: torch.Tensor,
    sampling_metadata: Any,
    *,
    target_token_id_candidates: torch.Tensor | None = None,
    relaxed_draft_mask: torch.Tensor | None = None,
    target_rows_selected: bool = False,
    bonus_token_ids: torch.Tensor | None = None,
) -> SamplerOutput:
    """Verify drafts against compact-vocabulary Target token IDs."""
    from vllm_ascend.sample import rejection_sampler as ascend_rejection

    if not sampling_metadata.all_greedy:
        raise ValueError("precomputed target token IDs require greedy sampling")
    if target_token_ids.ndim != 1:
        raise ValueError("target token IDs must be one-dimensional")

    target_argmax = (
        target_token_ids
        if target_rows_selected
        else target_token_ids[metadata.target_logits_indices]
    )
    if target_token_id_candidates is not None:
        if (
            target_token_id_candidates.ndim != 2
            or target_token_id_candidates.shape[0] != target_token_ids.shape[0]
        ):
            raise ValueError("target token candidates must have shape [num_logits, k]")
        target_candidates = target_token_id_candidates[metadata.target_logits_indices]
        draft_token_ids = metadata.draft_token_ids.to(target_candidates.dtype)
        relaxed_match = (target_candidates == draft_token_ids.unsqueeze(-1)).any(dim=-1)
        if relaxed_draft_mask is not None:
            if relaxed_draft_mask.shape != relaxed_match.shape:
                raise ValueError("relaxed draft mask must have shape [num_draft_tokens]")
            relaxed_match &= relaxed_draft_mask.to(
                device=target_candidates.device,
                dtype=torch.bool,
            )
        target_argmax = torch.where(
            relaxed_match,
            draft_token_ids,
            target_argmax,
        )
    elif relaxed_draft_mask is not None:
        if relaxed_draft_mask.shape != target_argmax.shape:
            raise ValueError("relaxed draft mask must have shape [num_draft_tokens]")
        draft_token_ids = metadata.draft_token_ids.to(target_argmax.dtype)
        target_argmax = torch.where(
            relaxed_draft_mask.to(
                device=target_argmax.device,
                dtype=torch.bool,
            ),
            draft_token_ids,
            target_argmax,
        )

    target_argmax = target_argmax.contiguous()
    if bonus_token_ids is None:
        if target_rows_selected:
            raise ValueError("selected target rows require explicit bonus token IDs")
        bonus_token_ids = target_token_ids[metadata.bonus_logits_indices]
    if bonus_token_ids.ndim == 1:
        bonus_token_ids = bonus_token_ids.unsqueeze(-1)
    bonus_token_ids = bonus_token_ids.to(torch.int32).contiguous()
    batch_size = len(metadata.num_draft_tokens)
    output_token_ids = torch.full(
        (batch_size, metadata.max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=target_token_ids.device,
    )
    if ascend_rejection.HAS_TRITON:
        grid, block_size = ascend_rejection.cal_grid_and_block_size(batch_size)
        ascend_rejection.rejection_greedy_sample_with_triton(
            output_token_ids,
            metadata.num_draft_tokens,
            metadata.cu_num_draft_tokens,
            metadata.draft_token_ids,
            target_argmax,
            bonus_token_ids,
            None,
            metadata.max_spec_len,
            grid,
            block_size,
        )
    elif min(metadata.num_draft_tokens) == 1 and max(metadata.num_draft_tokens) == 1:
        ascend_rejection.rejection_greedy_sample_spec_len_1_pytorch(
            output_token_ids,
            metadata.draft_token_ids,
            target_argmax,
            bonus_token_ids,
        )
    else:
        ascend_rejection.rejection_greedy_sample_pytorch(
            output_token_ids,
            metadata.cu_num_draft_tokens,
            metadata.draft_token_ids,
            target_argmax,
            bonus_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
        )
    self.last_verify_results = None
    return SamplerOutput(
        sampled_token_ids=output_token_ids,
        logprobs_tensors=None,
    )


def _trace_exact_topk_eligibility(owner: Any, sampling_metadata: Any) -> None:
    if os.environ.get("HUST_VSPEC_DRAFT_EXACT_REPETITION_TRACE") != "1":
        return
    if getattr(owner, "_vspec_topk_repetition_eligibility_traced", False):
        return
    owner._vspec_topk_repetition_eligibility_traced = True
    logits_processors = getattr(sampling_metadata, "logitsprocs", None)
    non_argmax_count = (
        -1
        if logits_processors is None
        else len(logits_processors.non_argmax_invariant)
    )
    print(
        "vSpec exact repetition eligibility: "
        f"method={os.environ.get('HUST_VSPEC_METHOD')} "
        f"all_greedy={sampling_metadata.all_greedy} "
        f"max_num_logprobs={sampling_metadata.max_num_logprobs} "
        f"logprob_token_ids={bool(sampling_metadata.logprob_token_ids)} "
        f"no_penalties={sampling_metadata.no_penalties} "
        f"repetition_only={getattr(sampling_metadata, '_vspec_repetition_only', None)} "
        f"prompt_token_ids={sampling_metadata.prompt_token_ids is not None} "
        f"spec_token_ids={sampling_metadata.spec_token_ids is not None} "
        f"allowed_mask={sampling_metadata.allowed_token_ids_mask is not None} "
        f"bad_words={bool(sampling_metadata.bad_words_token_ids)} "
        f"non_argmax_processors={non_argmax_count}",
        flush=True,
    )


def apply_linear_rejection_patch() -> bool:
    from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler

    _install_repetition_sampling_metadata_patch()
    if getattr(AscendRejectionSampler, PATCH_MARKER, False):
        return False
    AscendRejectionSampler.forward = _forward_linear_eagle
    AscendRejectionSampler.forward_greedy_token_ids = _forward_greedy_token_ids
    setattr(AscendRejectionSampler, PATCH_MARKER, True)
    return True


def _install_repetition_sampling_metadata_patch() -> bool:
    """Expose CPU-side penalty composition without synchronizing the NPU."""
    from vllm.v1.worker.gpu_input_batch import InputBatch

    if getattr(InputBatch, SAMPLING_METADATA_PATCH_MARKER, False):
        return False
    original_make_sampling_metadata = InputBatch._make_sampling_metadata

    def make_sampling_metadata(self: Any) -> Any:
        metadata = original_make_sampling_metadata(self)
        active_repetition_penalties = self.repetition_penalties_cpu[: self.num_reqs]
        metadata._vspec_repetition_only = (
            not self.no_penalties
            and not self.presence_penalties_reqs
            and not self.frequency_penalties_reqs
            and bool(self.repetition_penalties_reqs)
            and all(float(value) >= 1.0 for value in active_repetition_penalties)
        )
        metadata._vspec_req_ids = tuple(self.req_ids[: self.num_reqs])
        metadata._vspec_prompt_token_ids_cpu = self.token_ids_cpu
        metadata._vspec_prompt_lengths_cpu = self.num_prompt_tokens
        metadata._vspec_max_num_reqs = self.max_num_reqs
        return metadata

    InputBatch._make_sampling_metadata = make_sampling_metadata
    setattr(InputBatch, SAMPLING_METADATA_PATCH_MARKER, True)
    return True
