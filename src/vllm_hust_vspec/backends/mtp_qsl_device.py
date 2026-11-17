"""Opt-in Q35 MTP draft attention query-start-location device snapshot.

This is intentionally confined to the single-group, uniform Q35 MTP draft
path.  The host builder copies the CPU query starts to a fresh NPU tensor for
every draft step, although the proposer has already written the same values
to a device buffer.  Cloning that buffer preserves the host builder's private
tensor lifetime across asynchronous scheduler steps without the CPU pin/H2D.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch


_MARKER = "_vspec_q35_draft_qsl_device_snapshot"
_GAMMA_MARKER = "_vspec_q35_draft_qsl_gamma"
_OWNER_MARKER = "_vspec_q35_draft_qsl_owner"
_ABSENT = object()
_PROPOSAL_SCOPE: ContextVar[tuple[int, int] | None] = ContextVar(
    "vspec_q35_draft_qsl_proposal_scope", default=None
)
_active_calls = 0
_marked_calls = 0
_parity_confirmed_keys: set[tuple[int, bool]] = set()


def _host_flags_are_pure_decode(common: Any) -> bool:
    """Reject unknown or mixed prefill state without touching NPU data."""
    num_reqs = getattr(common, "num_reqs", None)
    flags = getattr(common, "is_prefilling", None)
    if not (
        type(num_reqs) is int
        and 0 < num_reqs <= 16
        and isinstance(flags, torch.Tensor)
        and flags.device.type == "cpu"
        and flags.dtype == torch.bool
        and flags.ndim == 1
        and flags.stride(0) == 1
        and flags.shape[0] >= num_reqs
    ):
        return False
    return not bool(flags[:num_reqs].any().item())


def _q35_single_group_proposer(
    proposer: Any,
    *,
    runtime_gamma: int | None = None,
    eagle_cls: type | None = None,
    capturing: bool | None = None,
) -> bool:
    if eagle_cls is None:
        from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

        eagle_cls = AscendEagleProposer
    if capturing is None:
        # This guard also runs at the outer _propose entry, before Ascend has
        # installed a forward context.  _EXTRA_CTX.capturing raises there;
        # the NPU stream probe explicitly supports an uninitialized context.
        # If the probe is unavailable, fail closed to the original builder.
        try:
            capturing = bool(torch.npu.is_current_stream_capturing())
        except Exception:
            return False

    config = getattr(proposer, "vllm_config", None)
    spec = getattr(config, "speculative_config", None)
    model_config = getattr(config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_types = (
        getattr(hf_config, "model_type", None),
        getattr(hf_text_config, "model_type", None),
    )
    return bool(
        type(proposer) is eagle_cls
        and getattr(spec, "method", None) == "mtp"
        and not capturing
        and any(str(value).startswith("qwen3_5") for value in model_types)
        and (
            runtime_gamma
            if runtime_gamma is not None
            else getattr(proposer, "num_speculative_tokens", None)
        ) in (4, 6, 8, 10, 12)
        and len(getattr(proposer, "draft_attn_groups", ())) == 1
        and not getattr(proposer, "parallel_drafting", False)
        and not getattr(proposer, "use_compress", False)
        and getattr(proposer, "dcp_size", 1) == 1
        and getattr(getattr(proposer, "runner", None), "dcp_manager", None) is None
    )


def _eligible_proposal_scope(
    proposer: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    eagle_cls: type | None = None,
    capturing: bool | None = None,
) -> int | None:
    """Only the runner's explicit pure-decode call can arm this draft path.

    Positional/legacy calls and absent batch facts fail closed. The adaptive
    wrapper may later execute with a different gamma; the draft/build guards
    recheck the actual gamma before using the snapshot.
    """
    required = (
        "num_speculative_tokens",
        "num_prefill_reqs",
        "num_decode_reqs",
        "target_model_batch_desc",
        "common_attn_metadata",
        "scheduler_output",
    )
    if args or any(key not in kwargs for key in required):
        return None
    gamma = kwargs["num_speculative_tokens"]
    prefill = kwargs["num_prefill_reqs"]
    decode = kwargs["num_decode_reqs"]
    desc = kwargs["target_model_batch_desc"]
    scheduler = kwargs["scheduler_output"]
    if not (
        type(gamma) is int
        and gamma in (4, 6, 8, 10, 12)
        and type(prefill) is int
        and prefill == 0
        and type(decode) is int
        # Non-DCP runner uses zero as a placeholder for both counters;
        # actual pure-decode classification comes from BatchDescriptor.uniform.
        and decode == 0
        and desc is not None
        and getattr(desc, "uniform", None) is True
        and kwargs["common_attn_metadata"] is not None
        and _host_flags_are_pure_decode(kwargs["common_attn_metadata"])
        and scheduler is not None
        and type(getattr(scheduler, "num_spec_tokens_to_schedule", None)) is int
        and scheduler.num_spec_tokens_to_schedule == gamma
    ):
        return None
    if not _q35_single_group_proposer(
        proposer,
        runtime_gamma=gamma,
        eagle_cls=eagle_cls,
        capturing=capturing,
    ):
        return None
    return gamma


def _scope_matches(owner: int, gamma: int) -> bool:
    return _PROPOSAL_SCOPE.get() == (owner, gamma)


def _eligible_builder(builder: Any, common: Any) -> bool:
    if not getattr(common, _MARKER, False):
        return False
    if getattr(common, _GAMMA_MARKER, -1) not in (4, 6, 8, 10, 12):
        return False
    if not _scope_matches(
        getattr(common, _OWNER_MARKER, -1), getattr(common, _GAMMA_MARKER, -1)
    ):
        return False
    from vllm.v1.kv_cache_interface import CrossAttentionSpec
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    if type(builder) is not AscendAttentionMetadataBuilder:
        return False
    if (
        builder.pcp_enabled
        or isinstance(builder.kv_cache_spec, CrossAttentionSpec)
        or getattr(builder.speculative_config, "parallel_drafting", False)
        or common.attn_state is not AscendAttentionState.SpecDecoding
        or common.causal is not True
        or not _host_flags_are_pure_decode(common)
    ):
        return False
    num_reqs = common.num_reqs
    cpu = common.query_start_loc_cpu
    device = common.query_start_loc
    seq_cpu = common._seq_lens_cpu
    if seq_cpu is None:
        seq_cpu = common.seq_lens_cpu
    block_table = common.block_table_tensor
    return bool(
        isinstance(num_reqs, int)
        and 0 < num_reqs <= 16
        and cpu is not None
        and device is not None
        and seq_cpu is not None
        and block_table is not None
        and cpu.device.type == "cpu"
        and seq_cpu.device.type == "cpu"
        and device.device.type == "npu"
        and block_table.device.type == "npu"
        and cpu.ndim == device.ndim == seq_cpu.ndim == 1
        and block_table.ndim == 2
        and cpu.dtype == device.dtype
        and cpu.stride(0) == device.stride(0) == 1
        and cpu.shape[0] >= num_reqs + 1
        and device.shape[0] >= num_reqs + 1
        and seq_cpu.shape[0] >= num_reqs
        and block_table.shape[0] >= num_reqs
    )


def _build_with_device_query_snapshot(builder: Any, common: Any) -> Any:
    """Match pinned host builder's non-PCP/no-padding branch exactly."""
    num_reqs = common.num_reqs
    num_actual_tokens = common.num_actual_tokens
    query_start_loc_cpu = common.query_start_loc_cpu[: num_reqs + 1]
    num_decodes, num_prefills, num_decode_tokens, _ = builder._split_decodes_and_prefills(common)

    seq_lens_source = common._seq_lens_cpu
    if seq_lens_source is None:
        seq_lens_source = common.seq_lens_cpu
    seq_lens = seq_lens_source[:num_reqs]
    slot_mapping = common.slot_mapping[:num_actual_tokens]
    attn_mask = builder.attn_mask_builder.get_attention_mask(common.causal, builder.model_config)
    # The original builder's pinned-CPU H2D creates a private device tensor.
    # A D2D clone is also private, preserving the async-lifetime guarantee.
    query_start_loc = common.query_start_loc[: num_reqs + 1].clone()
    actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
    seq_lens_list = seq_lens.tolist()
    backend_metadata = builder._build_backend_metadata(
        common,
        block_table=common.block_table_tensor,
        query_lens=query_start_loc_cpu[1:] - query_start_loc_cpu[:-1],
        seq_lens=seq_lens,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
    )
    return builder.metadata_cls(
        num_actual_tokens=num_actual_tokens,
        num_decode_tokens=num_decode_tokens,
        block_tables=common.block_table_tensor,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        seq_lens_list=seq_lens_list,
        max_query_len=common.max_query_len,
        actual_seq_lengths_q=actual_seq_lengths_q,
        slot_mapping=slot_mapping,
        attn_mask=attn_mask,
        attn_state=common.attn_state,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        causal=common.causal,
        model_runner_type=builder.model_config.runner_type,
        **backend_metadata,
    )


def apply_q35_draft_qsl_device_patch() -> bool:
    from vllm.logger import logger
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    original_build = AscendAttentionMetadataBuilder.build
    original_draft_build = AscendSpecDecodeBaseProposer.build_draft_attn_metadata
    original_propose = AscendSpecDecodeBaseProposer._propose
    if getattr(original_build, _MARKER, False):
        return False

    @wraps(original_propose)
    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        gamma = _eligible_proposal_scope(self, args, kwargs)
        if gamma is None:
            return original_propose(self, *args, **kwargs)
        token = _PROPOSAL_SCOPE.set((id(self), gamma))
        try:
            return original_propose(self, *args, **kwargs)
        finally:
            _PROPOSAL_SCOPE.reset(token)

    @wraps(original_draft_build)
    def build_draft_attn_metadata(
        self: Any, common_attn_metadata: Any, num_input_tokens: int, num_actual_tokens: int
    ) -> Any:
        global _marked_calls
        gamma = getattr(self, "num_speculative_tokens", None)
        eligible = bool(
            type(gamma) is int
            and _scope_matches(id(self), gamma)
            and _q35_single_group_proposer(self)
        )
        old_markers = {
            name: getattr(common_attn_metadata, name, _ABSENT)
            for name in (_MARKER, _GAMMA_MARKER, _OWNER_MARKER)
        }
        setattr(common_attn_metadata, _MARKER, eligible)
        setattr(common_attn_metadata, _GAMMA_MARKER, gamma if eligible else -1)
        setattr(common_attn_metadata, _OWNER_MARKER, id(self) if eligible else -1)
        if eligible:
            _marked_calls += 1
            if _marked_calls in (1, 2, 3):
                logger.warning(
                    "vSpec Q35 draft QSL device-snapshot marked call=%d gamma=%d",
                    _marked_calls,
                    self.num_speculative_tokens,
                )
        try:
            return original_draft_build(
                self, common_attn_metadata, num_input_tokens, num_actual_tokens
            )
        finally:
            for name, previous in old_markers.items():
                if previous is _ABSENT:
                    delattr(common_attn_metadata, name)
                else:
                    setattr(common_attn_metadata, name, previous)

    @wraps(original_build)
    def build(self: Any, common_prefix_len: int, common_attn_metadata: Any, *args: Any, **kwargs: Any) -> Any:
        global _active_calls
        if _eligible_builder(self, common_attn_metadata):
            parity_key = (
                int(getattr(common_attn_metadata, _GAMMA_MARKER, -1)),
                common_attn_metadata.max_query_len > 1,
            )
            if parity_key not in _parity_confirmed_keys:
                num_reqs = common_attn_metadata.num_reqs
                expected = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
                actual = common_attn_metadata.query_start_loc[: num_reqs + 1].cpu()
                if not torch.equal(actual, expected):
                    logger.warning(
                        "vSpec Q35 draft QSL device-snapshot parity mismatch; "
                        "falling back to host builder for this call"
                    )
                    return original_build(
                        self, common_prefix_len, common_attn_metadata, *args, **kwargs
                    )
                _parity_confirmed_keys.add(parity_key)
                logger.warning(
                    "vSpec Q35 draft QSL device-snapshot parity confirmed "
                    "gamma=%d full_window=%s",
                    parity_key[0],
                    parity_key[1],
                )
            result = _build_with_device_query_snapshot(self, common_attn_metadata)
            _active_calls += 1
            if _active_calls in (1, 2, 3, 128):
                logger.warning(
                    "vSpec Q35 draft QSL device-snapshot active call=%d num_reqs=%d",
                    _active_calls,
                    common_attn_metadata.num_reqs,
                )
            return result
        return original_build(self, common_prefix_len, common_attn_metadata, *args, **kwargs)

    setattr(build, _MARKER, True)
    setattr(build_draft_attn_metadata, _MARKER, True)
    setattr(propose, _MARKER, True)
    AscendAttentionMetadataBuilder.build = build
    AscendSpecDecodeBaseProposer.build_draft_attn_metadata = build_draft_attn_metadata
    AscendSpecDecodeBaseProposer._propose = propose
    logger.warning("vSpec Q35 draft QSL device-snapshot scoped patch installed (opt-in)")
    return True
