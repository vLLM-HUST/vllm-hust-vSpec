"""Private, opt-in Qwen3.5 MTP GDN metadata sharing experiment.

The first GDN group builds normally.  Later groups in the same target
metadata call reuse only its group-independent pure-spec graph metadata.
Each later group still aligns its own block table and refreshes its own
graph-stable tensors.  No device value is read by this patch.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from functools import wraps
import hashlib
import os
from pathlib import Path
from typing import Any

import torch

_RUNNER_SHA256 = "92610fb59add75e448f99c0e80421e9999168987e83eb2312c1c87f15eebbc72"
_GDN_SHA256 = "b50713c69ab65ee476a9d8a3ac15eb171faff0fea399e438cc7005e90a3dd57a"
_MARKER = "_vspec_q35_mtp_gdn_share"


@dataclass
class _Scope:
    builders: tuple[Any, Any, Any]
    runner: Any | None = None
    position: int = 0
    leader_metadata: Any | None = None
    leader_inputs: tuple[Any, ...] | None = None
    leader_indices: torch.Tensor | None = None
    disabled: bool = False
    hits: int = 0
    validated_hits: int = 0


_scope: ContextVar[_Scope | None] = ContextVar("vspec_q35_mtp_gdn_share_scope", default=None)


def _pinned_method(method: Any, filename: str, sha256: str) -> bool:
    if getattr(method, "__wrapped__", None) is not None:
        return False
    try:
        path = Path(method.__code__.co_filename).resolve()
        return path.name == filename and hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    except (AttributeError, OSError, TypeError):
        return False


def _q35_mtp_runner(runner: Any) -> bool:
    config = getattr(runner, "vllm_config", None)
    speculative = getattr(config, "speculative_config", None)
    model = getattr(config, "model_config", None)
    hf = getattr(model, "hf_config", None)
    hf_text = getattr(model, "hf_text_config", None)
    cache = getattr(config, "cache_config", None)
    return bool(
        getattr(speculative, "method", None) == "mtp"
        and getattr(speculative, "num_speculative_tokens", None) == 6
        and any(
            str(getattr(item, "model_type", "")).startswith("qwen3_5")
            for item in (hf, hf_text)
        )
        and getattr(cache, "mamba_cache_mode", None) == "align"
        and getattr(runner, "use_async_scheduling", False)
    )


def _target_builders(runner: Any, gdn_class: type[Any]) -> tuple[Any, Any, Any] | None:
    """Use the runner's actual builders; never infer groups from call count."""
    cached = getattr(runner, "_vspec_q35_mtp_gdn_share_builders", None)
    if cached is not None:
        return cached
    try:
        found = []
        for groups in runner.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder(0)
                if isinstance(builder, gdn_class):
                    found.append(builder)
        if len(found) != 3 or len({id(builder) for builder in found}) != 3:
            return None
        first = found[0]
        spec = first.kv_cache_spec
        if not all(
            type(builder) is gdn_class
            and builder.use_full_cuda_graph
            and builder.use_spec_decode
            and builder.num_spec == 6
            and builder.kv_cache_spec.block_size == spec.block_size
            and builder.kv_cache_spec.num_speculative_blocks == spec.num_speculative_blocks
            and builder.vllm_config is runner.vllm_config
            for builder in found
        ):
            return None
        result = tuple(found)
        runner._vspec_q35_mtp_gdn_share_builders = result
        return result
    except (AttributeError, TypeError, ValueError):
        return None


def _tensor_key(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _input_key(common: Any, accepted: Any, draft_cpu: Any) -> tuple[Any, ...] | None:
    """Only share when every group sees the same live input buffers."""
    if (
        not isinstance(common.query_start_loc_cpu, torch.Tensor)
        or common.query_start_loc_cpu.device.type != "cpu"
        or not isinstance(draft_cpu, torch.Tensor)
        or draft_cpu.device.type != "cpu"
    ):
        return None
    fields = (
        common.query_start_loc,
        common.query_start_loc_cpu,
        common.seq_lens,
        common.seq_lens_cpu_upper_bound,
        common.is_prefilling,
        accepted,
        draft_cpu,
    )
    keys = tuple(_tensor_key(value) for value in fields)
    if any(key is None for key in keys):
        return None
    return (common.num_reqs, common.num_actual_tokens, *keys)


def _pure_spec_graph(metadata: Any, common: Any, leader: Any) -> bool:
    """Require the exact graph-stable outputs produced by the pinned builder."""
    try:
        num_reqs = common.num_reqs
        num_spec = metadata.num_spec_decodes
        spec_decode = metadata.spec_decode_metadata
        return bool(
            type(num_reqs) is int
            and 0 < num_spec <= num_reqs <= leader.decode_cudagraph_max_bs
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0
            and metadata.num_spec_decode_tokens <= leader.decode_cudagraph_max_bs
            and metadata.num_actual_tokens == common.num_actual_tokens
            and metadata.spec_state_indices_tensor.shape == (num_reqs, leader.num_spec + 1)
            and metadata.spec_state_indices_tensor.data_ptr() == leader.spec_state_indices_tensor.data_ptr()
            and metadata.spec_sequence_masks.shape == (num_reqs,)
            and metadata.spec_sequence_masks.data_ptr() == leader.spec_sequence_masks.data_ptr()
            and metadata.spec_query_start_loc.shape == (num_reqs + 1,)
            and metadata.spec_query_start_loc.data_ptr() == leader.spec_query_start_loc.data_ptr()
            and metadata.num_accepted_tokens.shape == (num_reqs,)
            and metadata.num_accepted_tokens.data_ptr() == leader.num_accepted_tokens.data_ptr()
            and metadata.spec_token_indx.shape == (metadata.num_spec_decode_tokens,)
            and metadata.spec_token_indx.data_ptr() == leader.spec_token_indx.data_ptr()
            and metadata.non_spec_token_indx.numel() == 0
            and spec_decode.actual_seq_lengths.shape == (num_reqs + 1,)
            and spec_decode.actual_seq_lengths.data_ptr() == leader.spec_actual_seq_lengths.data_ptr()
            and leader.spec_sequence_indices[:num_spec].shape == (num_spec,)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _build_follower(
    builder: Any,
    common: Any,
    source: Any,
    leader_indices: torch.Tensor,
    *,
    align_block_table: Any,
    spec_decode_cls: Any,
    spec_conv_cls: Any,
) -> Any:
    """Refresh every group-specific/stable output of the pure-spec branch."""
    num_reqs = common.num_reqs
    num_spec = source.num_spec_decodes
    block_table = align_block_table(
        common.block_table_tensor,
        common.seq_lens,
        builder.kv_cache_spec,
        builder.vllm_config.cache_config.mamba_cache_mode,
    )
    state_snapshot = torch.index_select(
        block_table[:, : builder.num_spec + 1], 0, leader_indices[:num_spec]
    )

    # The original FULL-graph branch clears the entire tail, including rows
    # outside the current graph batch, before refreshing its stable inputs.
    builder.spec_state_indices_tensor[num_reqs:].fill_(0)  # NULL_BLOCK_ID
    builder.spec_state_indices_tensor[:num_spec].copy_(state_snapshot, non_blocking=True)
    state_indices = builder.spec_state_indices_tensor[:num_reqs]
    state_indices[num_spec:].fill_(0)

    sequence_masks = builder.spec_sequence_masks[:num_reqs]
    sequence_masks.copy_(source.spec_sequence_masks, non_blocking=True)
    sequence_masks[num_spec:].fill_(False)

    token_indices = builder.spec_token_indx[: source.spec_token_indx.size(0)]
    token_indices.copy_(source.spec_token_indx, non_blocking=True)
    non_spec_indices = builder.non_spec_token_indx[:0]

    query_start_loc = builder.spec_query_start_loc[: num_reqs + 1]
    query_start_loc.copy_(source.spec_query_start_loc, non_blocking=True)
    accepted = builder.num_accepted_tokens[:num_reqs]
    accepted.copy_(source.num_accepted_tokens, non_blocking=True)
    actual_lengths = builder.spec_actual_seq_lengths[: num_reqs + 1]
    actual_lengths.copy_(source.spec_decode_metadata.actual_seq_lengths, non_blocking=True)

    result = replace(
        source,
        spec_query_start_loc=query_start_loc,
        spec_state_indices_tensor=state_indices,
        spec_sequence_masks=sequence_masks,
        spec_token_indx=token_indices,
        non_spec_token_indx=non_spec_indices,
        num_accepted_tokens=accepted,
    )
    result.spec_decode_metadata = spec_decode_cls(
        spec_causal_conv1d=spec_conv_cls(
            query_start_loc=query_start_loc,
            cache_indices=state_indices,
            num_accepted_tokens=accepted,
        ),
        actual_seq_lengths=actual_lengths,
    )
    result.non_spec_prefill_metadata = None
    result.non_spec_decode_metadata = None
    return result


def _diagnostic_gamma(draft_cpu: Any, num_reqs: int) -> int | None:
    """Identify uniform dynamic gamma from the existing host draft-width input."""
    if (
        not isinstance(draft_cpu, torch.Tensor)
        or draft_cpu.device.type != "cpu"
        or draft_cpu.ndim != 1
        or draft_cpu.numel() != num_reqs
    ):
        return None
    positive = {int(value) for value in draft_cpu.tolist() if int(value) > 0}
    return next(iter(positive)) if len(positive) == 1 and positive <= {4, 6} else None


def _diagnostic_log_once(scope: _Scope, group: int, gamma: int, status: str, logger: Any) -> None:
    runner = scope.runner
    assert runner is not None
    seen = getattr(runner, "_vspec_q35_gdn_share_diag_seen", None)
    if seen is None:
        seen = set()
        runner._vspec_q35_gdn_share_diag_seen = seen
    key = (group, gamma, status)
    if key in seen:
        return
    seen.add(key)
    if group == 0:
        logger.warning(
            "Q35 GDN share diagnostic: pid=%d, group=0, role=leader, gamma=%d, status=eligible",
            os.getpid(), gamma,
        )
    else:
        logger.warning(
            "Q35 GDN share diagnostic: pid=%d, group=%d, role=follower, gamma=%d, "
            "status=parity_pass, state=1, tail=1, masks=1, tokens=1, query=1, accepted=1, actual=1",
            os.getpid(), group, gamma,
        )


def _validate_follower(
    builder: Any,
    candidate: Any,
    original_build: Any,
    original_args: tuple[Any, ...],
    gamma: int,
    group: int,
) -> Any:
    """Diagnostic only: compare saved optimized outputs with a fresh original build.

    The original build rewrites the same graph-stable buffers, so every
    candidate tensor must be cloned *before* it runs.  Returning the original
    metadata keeps the safety smoke on the known-good result after parity.
    Device comparisons synchronize and are forbidden in throughput runs.
    """
    num_reqs = original_args[1].num_reqs
    values = {
        "state": candidate.spec_state_indices_tensor.clone(),
        "full_state": builder.spec_state_indices_tensor.clone(),
        "masks": candidate.spec_sequence_masks.clone(),
        "tokens": candidate.spec_token_indx.clone(),
        "non_spec_tokens": candidate.non_spec_token_indx.clone(),
        "query": candidate.spec_query_start_loc.clone(),
        "accepted": candidate.num_accepted_tokens.clone(),
        "actual": candidate.spec_decode_metadata.actual_seq_lengths.clone(),
    }
    baseline = original_build(builder, *original_args)
    checks = {
        "state": torch.equal(values["state"], baseline.spec_state_indices_tensor),
        "full_state": torch.equal(values["full_state"], builder.spec_state_indices_tensor),
        "tail": torch.equal(
            values["full_state"][num_reqs:],
            torch.zeros_like(values["full_state"][num_reqs:]),
        ),
        "masks": torch.equal(values["masks"], baseline.spec_sequence_masks),
        "tokens": torch.equal(values["tokens"], baseline.spec_token_indx),
        "non_spec_tokens": torch.equal(values["non_spec_tokens"], baseline.non_spec_token_indx),
        "query": torch.equal(values["query"], baseline.spec_query_start_loc),
        "accepted": torch.equal(values["accepted"], baseline.num_accepted_tokens),
        "actual": torch.equal(values["actual"], baseline.spec_decode_metadata.actual_seq_lengths),
    }
    for field in fields(candidate):
        left = getattr(candidate, field.name)
        right = getattr(baseline, field.name)
        if isinstance(left, torch.Tensor):
            if field.name not in {
                "spec_state_indices_tensor", "spec_sequence_masks", "spec_token_indx",
                "non_spec_token_indx", "spec_query_start_loc", "num_accepted_tokens",
            }:
                checks[f"unexpected_field_{field.name}"] = False
        else:
            checks[f"field_{field.name}"] = left == right
    checks["spec_conv_state"] = torch.equal(
        values["state"], baseline.spec_decode_metadata.spec_causal_conv1d.cache_indices
    )
    checks["spec_conv_query"] = torch.equal(
        values["query"], baseline.spec_decode_metadata.spec_causal_conv1d.query_start_loc
    )
    checks["spec_conv_accepted"] = torch.equal(
        values["accepted"], baseline.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens
    )
    checks["non_spec_attachments"] = (
        candidate.non_spec_prefill_metadata is None
        and candidate.non_spec_decode_metadata is None
        and baseline.non_spec_prefill_metadata is None
        and baseline.non_spec_decode_metadata is None
    )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"Q35 GDN share parity mismatch group={group} gamma={gamma} fields={failed}"
        )
    return baseline


def apply_q35_mtp_gdn_share_patch() -> bool:
    from vllm.logger import logger
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder,
        GDNSpecCausalConv1dMetadata,
        GDNSpecDecodeMetadata,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    gdn_class = AscendGDNAttentionMetadataBuilder
    original_gdn = gdn_class.build
    original_runner = NPUModelRunner._build_attention_metadata
    if getattr(original_gdn, _MARKER, False):
        return False
    if not (
        _pinned_method(original_gdn, "gdn_attn_builder.py", _GDN_SHA256)
        and _pinned_method(original_runner, "model_runner_v1.py", _RUNNER_SHA256)
    ):
        logger.warning("vSpec Q35 MTP GDN sharing source changed; original builders retained")
        return False
    trace = os.getenv("HUST_ISSUE18_Q35_GDN_MTP_SHARE_TRACE", "0") == "1"
    validate = os.getenv("HUST_ISSUE18_Q35_GDN_MTP_SHARE_VALIDATE", "0") == "1"

    @wraps(original_runner)
    def target_metadata(self: Any, *args: Any, **kwargs: Any) -> Any:
        capture = kwargs.get("for_cudagraph_capture", args[8] if len(args) > 8 else False)
        use_spec = kwargs.get("use_spec_decode", args[7] if len(args) > 7 else False)
        if capture or not use_spec or not _q35_mtp_runner(self):
            return original_runner(self, *args, **kwargs)
        builders = _target_builders(self, gdn_class)
        if builders is None:
            return original_runner(self, *args, **kwargs)
        token = _scope.set(_Scope(builders, runner=self))
        try:
            result = original_runner(self, *args, **kwargs)
            if trace:
                scope = _scope.get()
                assert scope is not None
                calls = getattr(self, "_vspec_q35_mtp_gdn_share_calls", 0) + 1
                hits = getattr(self, "_vspec_q35_mtp_gdn_share_hits", 0) + scope.hits
                validated = getattr(self, "_vspec_q35_mtp_gdn_share_validated_hits", 0) + scope.validated_hits
                self._vspec_q35_mtp_gdn_share_calls = calls
                self._vspec_q35_mtp_gdn_share_hits = hits
                self._vspec_q35_mtp_gdn_share_validated_hits = validated
                if calls in (1, 16, 128, 1024):
                    logger.warning(
                        "vSpec Q35 MTP GDN sharing calls=%d follower_hits=%d "
                        "validated_hits=%d last_call_hits=%d",
                        calls, hits, validated, scope.hits,
                    )
            return result
        finally:
            _scope.reset(token)

    @wraps(original_gdn)
    def build(
        self: Any,
        common_prefix_len: int,
        common_attn_metadata: Any,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> Any:
        scope = _scope.get()
        if scope is None or scope.disabled or scope.position >= 3:
            return original_gdn(
                self, common_prefix_len, common_attn_metadata,
                num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build,
            )
        if self is not scope.builders[scope.position]:
            if any(self is builder for builder in scope.builders):
                scope.disabled = True
            return original_gdn(
                self, common_prefix_len, common_attn_metadata,
                num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build,
            )
        scope.position += 1
        inputs = _input_key(
            common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu
        )
        if scope.position == 1:
            result = original_gdn(
                self, common_prefix_len, common_attn_metadata,
                num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build,
            )
            if inputs is not None and _pure_spec_graph(result, common_attn_metadata, self):
                scope.leader_metadata = result
                scope.leader_inputs = inputs
                scope.leader_indices = self.spec_sequence_indices[:result.num_spec_decodes]
                if validate:
                    gamma = _diagnostic_gamma(num_decode_draft_tokens_cpu, common_attn_metadata.num_reqs)
                    if gamma is not None:
                        _diagnostic_log_once(scope, 0, gamma, "eligible", logger)
            else:
                scope.disabled = True
            return result
        if (
            fast_build
            or inputs is None
            or inputs != scope.leader_inputs
            or scope.leader_metadata is None
            or scope.leader_indices is None
            or not _pure_spec_graph(scope.leader_metadata, common_attn_metadata, scope.builders[0])
            or self.decode_cudagraph_max_bs != scope.builders[0].decode_cudagraph_max_bs
            or not self.use_full_cuda_graph
            or not self.use_spec_decode
            or self.num_spec != 6
            or self.kv_cache_spec.block_size != scope.builders[0].kv_cache_spec.block_size
            or self.kv_cache_spec.num_speculative_blocks != scope.builders[0].kv_cache_spec.num_speculative_blocks
        ):
            scope.disabled = True
            return original_gdn(
                self, common_prefix_len, common_attn_metadata,
                num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build,
            )
        gamma = None
        if validate:
            gamma = _diagnostic_gamma(num_decode_draft_tokens_cpu, common_attn_metadata.num_reqs)
            if gamma is None:
                scope.disabled = True
                return original_gdn(
                    self, common_prefix_len, common_attn_metadata,
                    num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build,
                )
        result = _build_follower(
            self,
            common_attn_metadata,
            scope.leader_metadata,
            scope.leader_indices,
            align_block_table=mamba_get_block_table_tensor,
            spec_decode_cls=GDNSpecDecodeMetadata,
            spec_conv_cls=GDNSpecCausalConv1dMetadata,
        )
        scope.hits += 1
        if validate:
            assert gamma in (4, 6)
            result = _validate_follower(
                self, result, original_gdn,
                (common_prefix_len, common_attn_metadata, num_accepted_tokens,
                 num_decode_draft_tokens_cpu, fast_build),
                gamma, scope.position - 1,
            )
            scope.validated_hits += 1
            _diagnostic_log_once(scope, scope.position - 1, gamma, "parity_pass", logger)
        return result

    setattr(build, _MARKER, True)
    setattr(target_metadata, _MARKER, True)
    NPUModelRunner._build_attention_metadata = target_metadata
    gdn_class.build = build
    logger.warning(
        "vSpec Q35 MTP GDN pure-spec sharing enabled: pid=%d (private opt-in)",
        os.getpid(),
    )
    return True
