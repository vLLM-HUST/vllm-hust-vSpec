"""HOLD: proposal-scoped single-launch Q35 Eagle-MTP metadata integration.

Private experimental code.  The CPU state-machine tests pass, but NPU Triton
lowering, graph replay and attention-builder parity have not been established.
The opt-in installer is never selected by the frozen source in the live run.
"""

from __future__ import annotations

import os
import logging
import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from itertools import count
from threading import Lock
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class UpdateCall:
    draft_index: int
    old_common_metadata: Any
    batch_size: int
    input_batch_size: int
    used_update_positions: torch.Tensor
    aclgraph_runtime_mode: Any
    ori_seq_len: Any
    ori_seq_len_cpu: Any
    slot_indices: Any
    mtp_slot_mapping: Any
    attn_group: Any

    def arguments(self) -> dict[str, Any]:
        return dict(
            draft_index=self.draft_index,
            old_common_metadata=self.old_common_metadata,
            batch_size=self.batch_size,
            input_batch_size=self.input_batch_size,
            used_update_positions=self.used_update_positions,
            attn_group=self.attn_group,
        )

    def extra_args_absent(self) -> bool:
        return all(
            value is None
            for value in (
                self.ori_seq_len, self.ori_seq_len_cpu,
                self.slot_indices, self.mtp_slot_mapping,
            )
        )


@dataclass
class BatchPlan:
    gamma: int
    full_mode: bool
    attention_state: Any
    padding_slot_id: int
    positions: list[torch.Tensor]
    used_identity: int
    group_identity: int
    batch_size: int
    input_batch_size: int
    runtime_mode: Any


@dataclass
class ProposalScope:
    owner: Any
    gamma: int
    proposal_id: int = 0
    mode: str = "pending"  # pending, serial, active, aborted
    next_index: int = 1
    expected_old: Any = None
    plan: BatchPlan | None = None
    shadow: Any = None
    actual_results: list[tuple[Any, Any]] | None = None
    shadow_checked: bool = False


_SCOPE: ContextVar[ProposalScope | None] = ContextVar(
    "vspec_q35_eagle_multistep_scope", default=None
)
_MARKER = "_vspec_q35_eagle_multistep_metadata"
_LOGGER = logging.getLogger(__name__)
_PROPOSAL_IDS = count(1)
_SHADOW_COUNTS = {4: 0, 6: 0}
_SHADOW_LOCK = Lock()


def _validation_enabled() -> bool:
    return os.getenv("HUST_ISSUE18_Q35_EAGLE_MULTISTEP_VALIDATE") == "1"


def _reserve_shadow(gamma: int) -> bool:
    """Sample only the first two qualified proposals per worker and gamma."""
    with _SHADOW_LOCK:
        count_so_far = _SHADOW_COUNTS[gamma]
        if count_so_far >= 2:
            return False
        _SHADOW_COUNTS[gamma] = count_so_far + 1
        return True


def _audit(event: str, scope: ProposalScope, step: int, detail: str) -> None:
    if not _validation_enabled():
        return
    _LOGGER.warning(
        "Q35MS3 event=%s pid=%d gamma=%s proposal=%d step=%d detail=%s",
        event, os.getpid(), scope.gamma, scope.proposal_id, step, detail,
    )


def make_proposal_wrappers(
    eagle_cls: type,
    original_propose: Callable[..., Any],
    original_update: Callable[..., Any],
    *,
    outer_eligible: Callable[[Any, tuple[Any, ...], dict[str, Any]], int | None],
    preflight: Callable[[Any, ProposalScope, UpdateCall], BatchPlan | None],
    launch: Callable[[Any, BatchPlan, UpdateCall], None],
    followup_valid: Callable[[Any, ProposalScope, UpdateCall], bool],
    materialize: Callable[[Any, BatchPlan, UpdateCall], tuple[Any, Any]],
    shadow_prepare: Callable[[Any, UpdateCall, int], Any] | None = None,
    shadow_compare: Callable[[Any, torch.Tensor, list[tuple[Any, Any]], Any], None] | None = None,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Injectable wrappers permit complete CPU/mocked failure-path tests."""

    @wraps(original_propose)
    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        gamma = outer_eligible(self, args, kwargs) if type(self) is eagle_cls else None
        if gamma is None:
            return original_propose(self, *args, **kwargs)
        scope = ProposalScope(owner=self, gamma=gamma, proposal_id=next(_PROPOSAL_IDS))
        token = _SCOPE.set(scope)
        try:
            result = original_propose(self, *args, **kwargs)
            if scope.mode == "aborted":
                raise RuntimeError("Q35 multistep proposal swallowed an abort after launch")
            if scope.mode == "active" and scope.next_index != gamma:
                scope.mode = "aborted"
                raise RuntimeError("Q35 multistep proposal returned before all continuation steps")
            if scope.shadow is not None and not scope.shadow_checked:
                scope.mode = "aborted"
                raise RuntimeError("Q35 shadow proposal returned without parity comparison")
            return result
        finally:
            _SCOPE.reset(token)

    @wraps(original_update)
    def update(
        self: Any,
        draft_index: int,
        old_common_metadata: Any,
        batch_size: int,
        input_batch_size: int,
        used_update_positions: torch.Tensor,
        aclgraph_runtime_mode: Any,
        ori_seq_len: Any = None,
        ori_seq_len_cpu: Any = None,
        slot_indices: Any = None,
        mtp_slot_mapping: Any = None,
        attn_group: Any = None,
    ) -> Any:
        call = UpdateCall(
            draft_index, old_common_metadata, batch_size, input_batch_size,
            used_update_positions, aclgraph_runtime_mode, ori_seq_len,
            ori_seq_len_cpu, slot_indices, mtp_slot_mapping, attn_group,
        )

        def serial() -> Any:
            return original_update(
                self, draft_index, old_common_metadata, batch_size,
                input_batch_size, used_update_positions, aclgraph_runtime_mode,
                ori_seq_len=ori_seq_len, ori_seq_len_cpu=ori_seq_len_cpu,
                slot_indices=slot_indices, mtp_slot_mapping=mtp_slot_mapping,
                attn_group=attn_group,
            )

        scope = _SCOPE.get()
        if scope is None or scope.owner is not self:
            return serial()
        if scope.mode == "serial":
            return serial()
        if scope.mode == "aborted":
            raise RuntimeError("Q35 multistep proposal already aborted")

        if scope.mode == "pending":
            if draft_index != 1:
                scope.mode = "serial"
                _audit("serial_fallback", scope, draft_index, "first_step_not_one")
                return serial()
            try:
                plan = preflight(self, scope, call)
            except Exception:
                if _validation_enabled():
                    scope.mode = "aborted"
                    _audit("shadow_fail", scope, draft_index, "preflight_exception")
                    raise
                # In normal mode no output has been written; serial fallback is safe.
                plan = None
            if plan is None:
                scope.mode = "serial"
                if _capturing() is True:
                    _audit("capture_reject", scope, draft_index, "capturing")
                else:
                    _audit("serial_fallback", scope, draft_index, "preflight_reject")
                return serial()
            scope.plan = plan
            if _validation_enabled() and _reserve_shadow(scope.gamma):
                if shadow_prepare is None or shadow_compare is None:
                    scope.mode = "aborted"
                    _audit("shadow_fail", scope, draft_index, "callbacks_missing")
                    raise RuntimeError("Q35 shadow validation callbacks missing")
                try:
                    scope.shadow = shadow_prepare(self, call, scope.gamma)
                    if scope.shadow is None:
                        raise RuntimeError("Q35 shadow preparation returned None")
                    scope.actual_results = []
                except Exception:
                    scope.mode = "aborted"
                    _audit("shadow_fail", scope, draft_index, "prepare")
                    raise
            try:
                launch(self, plan, call)
            except Exception:
                # Launch may have partially updated device buffers, including
                # used positions.  Silent host fallback would double-increment.
                scope.mode = "aborted"
                if scope.shadow is not None:
                    _audit("shadow_fail", scope, draft_index, "batch_launch")
                raise
            scope.mode = "active"
            _audit("batch_launch", scope, draft_index, "kernel_ok")
        else:
            if (scope.plan is None or draft_index != scope.next_index
                    or old_common_metadata is not scope.expected_old):
                scope.mode = "aborted"
                raise RuntimeError("Q35 multistep continuation order/identity changed after launch")
            try:
                valid = followup_valid(self, scope, call)
            except Exception:
                scope.mode = "aborted"
                raise
            if not valid:
                scope.mode = "aborted"
                raise RuntimeError("Q35 multistep continuation guard changed after launch")

        assert scope.plan is not None
        try:
            result = materialize(self, scope.plan, call)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("Q35 multistep materializer returned malformed result")
        except Exception:
            scope.mode = "aborted"
            if scope.shadow is not None:
                _audit("shadow_fail", scope, draft_index, "materialize")
            raise
        if scope.shadow is not None:
            assert scope.actual_results is not None and shadow_compare is not None
            scope.actual_results.append(result)
            if draft_index == scope.gamma - 1:
                try:
                    shadow_compare(self, call.used_update_positions,
                                   scope.actual_results, scope.shadow)
                except Exception as error:
                    scope.mode = "aborted"
                    detail = getattr(error, "field", "compare_exception")
                    _audit("shadow_fail", scope, draft_index, str(detail))
                    raise
                scope.shadow_checked = True
                _audit("shadow_pass", scope, draft_index, "all_fields")
        elif _validation_enabled() and draft_index == scope.gamma - 1:
            # The first two qualified proposals per worker/gamma were checked;
            # this later launch is explicitly *not* a parity claim.
            _audit("shadow_skip", scope, draft_index, "sample_budget")
        scope.expected_old = result[0]
        scope.next_index += 1
        return result

    setattr(propose, _MARKER, True)
    setattr(update, _MARKER, True)
    return propose, update


def _outer_eligible(proposer: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    """All facts available before `_propose`; later mode/buffers need step 1."""
    required = (
        "num_speculative_tokens", "num_prefill_reqs", "num_decode_reqs",
        "target_model_batch_desc", "common_attn_metadata", "scheduler_output",
    )
    if args or any(name not in kwargs for name in required):
        return None
    gamma = kwargs["num_speculative_tokens"]
    common = kwargs["common_attn_metadata"]
    flags = getattr(common, "is_prefilling", None)
    num_reqs = getattr(common, "num_reqs", None)
    scheduler = kwargs["scheduler_output"]
    if not (
        type(gamma) is int and gamma in (4, 6)
        and type(kwargs["num_prefill_reqs"]) is int
        and kwargs["num_prefill_reqs"] == 0
        and type(kwargs["num_decode_reqs"]) is int
        and kwargs["num_decode_reqs"] == 0
        and getattr(kwargs["target_model_batch_desc"], "uniform", None) is True
        and getattr(proposer, "_vspec_q35_metadata_fusion", False)
        and getattr(proposer, "method", None) == "mtp"
        and getattr(proposer, "uses_mrope", False)
        and type(getattr(scheduler, "num_spec_tokens_to_schedule", None)) is int
        and scheduler.num_spec_tokens_to_schedule == gamma
        and type(num_reqs) is int
        and 0 < num_reqs <= 16
        and isinstance(flags, torch.Tensor)
        and flags.device.type == "cpu"
        and flags.dtype == torch.bool
        and flags.ndim == 1
        and flags.numel() >= num_reqs
        and not bool(flags[:num_reqs].any().item())
    ):
        return None
    return gamma


def _capturing() -> bool | None:
    checker = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    if checker is None:
        return None
    try:
        return bool(checker())
    except Exception:
        return None


def _storage_id(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _safe_builder(group: Any) -> bool:
    """Only the inspected, non-PCP full-attention builder is precomputable."""
    from vllm.v1.kv_cache_interface import CrossAttentionSpec
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder

    builder = group.get_metadata_builder()
    return bool(
        type(builder) is AscendAttentionMetadataBuilder
        and not builder.pcp_enabled
        and not isinstance(builder.kv_cache_spec, CrossAttentionSpec)
        and not getattr(builder.speculative_config, "parallel_drafting", False)
    )


def _preflight(proposer: Any, scope: ProposalScope, call: UpdateCall) -> BatchPlan | None:
    from vllm.config import CUDAGraphMode
    from vllm_ascend.spec_decode.llm_base_proposer import AscendAttentionState, PADDING_SLOT_ID
    from .draft_metadata_fusion import can_fuse_q35_mtp_continuation

    capturing = _capturing()
    if capturing is not False or not call.extra_args_absent() or not _safe_builder(call.attn_group):
        return None
    full = call.aclgraph_runtime_mode == CUDAGraphMode.FULL
    piecewise = call.aclgraph_runtime_mode == CUDAGraphMode.PIECEWISE
    if not can_fuse_q35_mtp_continuation(
        proposer, call.arguments(), full_mode=full, piecewise_mode=piecewise,
        capturing=False, spec_decode_state=AscendAttentionState.SpecDecoding,
    ):
        return None
    gamma = scope.gamma
    if proposer.num_speculative_tokens != gamma:
        return None
    old = call.old_common_metadata
    groups = (proposer.slot_mapping_group, proposer.seq_lens_group,
              proposer.query_start_loc_group)
    if any(len(group) < gamma for group in groups):
        return None
    dest = [tensor for group in groups for tensor in group[1:gamma]]
    source = [call.used_update_positions, old.seq_lens, old.positions,
              old.query_start_loc, old.block_table_tensor]
    all_tensors = source + dest
    if len({_storage_id(t) for t in all_tensors}) != len(all_tensors):
        return None
    # The single kernel has one shared capacity per output kind.  Inactive
    # groups (dynamic gamma 4 in a max-gamma 6 pool) are intentionally ignored.
    for group in groups:
        first = group[1]
        if any(
            tensor.shape != first.shape
            or tensor.dtype != torch.int32
            or tensor.device != first.device
            or not tensor.is_contiguous()
            for tensor in group[1:gamma]
        ):
            return None
    positions = [torch.empty_like(old.positions) for _ in range(gamma - 1)]
    if len({_storage_id(t) for t in all_tensors + positions}) != len(all_tensors) + len(positions):
        return None
    return BatchPlan(
        gamma, full, AscendAttentionState.SpecDecoding, PADDING_SLOT_ID,
        positions, id(call.used_update_positions), id(call.attn_group),
        call.batch_size, call.input_batch_size, call.aclgraph_runtime_mode,
    )


def _launch(proposer: Any, plan: BatchPlan, call: UpdateCall) -> None:
    from .eagle_multistep_kernel import launch_multistep

    launch_multistep(
        proposer, call.old_common_metadata, call.used_update_positions,
        batch_size=call.batch_size, input_batch_size=call.input_batch_size,
        full_mode=plan.full_mode, padding_slot_id=plan.padding_slot_id,
        positions=plan.positions,
    )


def _followup_valid(proposer: Any, scope: ProposalScope, call: UpdateCall) -> bool:
    from .draft_metadata_fusion import can_fuse_q35_mtp_continuation

    plan = scope.plan
    if plan is None or _capturing() is not False:
        return False
    return bool(
        proposer.num_speculative_tokens == plan.gamma
        and id(call.used_update_positions) == plan.used_identity
        and id(call.attn_group) == plan.group_identity
        and _safe_builder(call.attn_group)
        and call.batch_size == plan.batch_size
        and call.input_batch_size == plan.input_batch_size
        and call.aclgraph_runtime_mode == plan.runtime_mode
        and call.extra_args_absent()
        and can_fuse_q35_mtp_continuation(
            proposer, call.arguments(), full_mode=plan.full_mode,
            piecewise_mode=not plan.full_mode, capturing=False,
            spec_decode_state=plan.attention_state,
        )
    )


def _materialize(proposer: Any, plan: BatchPlan, call: UpdateCall) -> tuple[Any, Any]:
    from .draft_metadata_fusion import fuse_q35_mtp_continuation

    return fuse_q35_mtp_continuation(
        proposer, call.arguments(), full_mode=plan.full_mode,
        attention_state=plan.attention_state,
        padding_slot_id=plan.padding_slot_id,
        precomputed_position=plan.positions[call.draft_index - 1],
        skip_kernel=True,
    )


def apply_q35_eagle_multistep_metadata_patch() -> bool:
    """Install only after the existing per-step Eagle fusion is installed."""
    base_enabled = (
        os.getenv("VSPEC_DRAFT_Q35_METADATA_FUSION") == "1"
        and os.getenv("HUST_ISSUE18_Q35_EAGLE_MULTISTEP") == "1"
    )
    if not base_enabled:
        if _validation_enabled():
            raise RuntimeError("Q35 shadow validation requested without both base opt-ins")
        return False
    from vllm.logger import logger
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

    original_propose = AscendEagleProposer._propose
    original_update = AscendEagleProposer.attn_update_stack_num_spec_norm
    if getattr(original_propose, _MARKER, False):
        if _validation_enabled() and not getattr(original_propose, "_vspec_q35_shadow_validation", False):
            raise RuntimeError("Q35 shadow validation requested after non-validation installer")
        return False
    if not getattr(original_update, "_vspec_q35_eagle_metadata_fusion", False):
        if _validation_enabled():
            raise RuntimeError("Q35 shadow validation requires installed per-step Eagle fusion")
        logger.warning("Q35 multistep installer HOLD: per-step Eagle fusion not installed")
        return False
    shadow_prepare = None
    shadow_compare = None
    if _validation_enabled():
        from .eagle_shadow_validation import (
            compare_shadow, prepare_shadow, resolve_native_updater,
        )

        # A changed upstream/compact wrapper is a hard installation error in
        # validation mode.  Never produce a smoke run without a proved oracle.
        native = resolve_native_updater()
        if inspect.unwrap(original_update) is not native:
            raise RuntimeError("Q35 shadow oracle differs from live per-step fallback updater")
        shadow_prepare = prepare_shadow
        shadow_compare = compare_shadow
    propose, update = make_proposal_wrappers(
        AscendEagleProposer, original_propose, original_update,
        outer_eligible=_outer_eligible,
        preflight=_preflight,
        launch=_launch,
        followup_valid=_followup_valid,
        materialize=_materialize,
        shadow_prepare=shadow_prepare,
        shadow_compare=shadow_compare,
    )
    AscendEagleProposer._propose = propose
    AscendEagleProposer.attn_update_stack_num_spec_norm = update
    if _validation_enabled():
        setattr(propose, "_vspec_q35_shadow_validation", True)
        setattr(update, "_vspec_q35_shadow_validation", True)
    logger.warning("Q35 Eagle multistep metadata prototype installed: NPU validation REQUIRED")
    return True
