"""Integration boundary for native Qwen3.5 MTP speculation."""

from __future__ import annotations

import logging
import os
import time
import traceback
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch

from ..config import PluginSettings
from . import mamba_compat

logger = logging.getLogger(__name__)


_DEVICE_COUNT_EVENT_OWNERS: dict[int, Any] = {}
_DEVICE_COUNT_PRECOPY_PATCH_MARKER = "_vllm_hust_vspec_device_count_precopy"
_GENERIC_MTP_LOCAL_ARGMAX: ContextVar[bool] = ContextVar(
    "vspec_generic_mtp_local_argmax", default=False
)


def _get_current_mamba_groups(kv_cache_config: Any) -> dict[Any, list[int]]:
    return mamba_compat.get_current_mamba_groups(kv_cache_config)


def _normalize_mamba_state_copy_funcs(
    kv_cache_config: Any,
    copy_funcs: Any,
) -> Any:
    return mamba_compat.normalize_mamba_state_copy_funcs(kv_cache_config, copy_funcs)


def _install_current_mamba_runtime_api() -> None:
    mamba_compat.install_current_mamba_runtime_api()


def _patch_mtp_mamba_group_api() -> bool:
    return mamba_compat.apply_mamba_runtime_compatibility_patch()


_REQUIRED_ASCEND_MTP_OPS = (
    "npu_gemma_rms_norm",
    "moe_gating_top_k",
    "npu_causal_conv1d_custom",
)


def _missing_ascend_mtp_ops(custom_namespace: Any) -> tuple[str, ...]:
    return tuple(name for name in _REQUIRED_ASCEND_MTP_OPS if not hasattr(custom_namespace, name))


def _validate_ascend_custom_ops() -> None:
    """Fail early when vLLM-Ascend and the active CANN runtime do not match."""
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    custom_namespace = getattr(torch.ops, "_C_ascend", None)
    missing = _missing_ascend_mtp_ops(custom_namespace)
    if missing:
        raise RuntimeError(
            "Qwen3.5 MTP requires native vLLM-Ascend custom ops that are not "
            f"available: {', '.join(missing)}. Source the CANN set_env.sh used "
            "to build the installed vLLM-Ascend extension before starting vSpec."
        )


def _patch_mtp_drafter_graph_guard() -> bool:
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )

    original_load_model = AscendSpecDecodeBaseProposer.load_model
    if getattr(original_load_model, "_vspec_mtp_strict_graph", False):
        return False

    @wraps(original_load_model)
    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        if self.method == "mtp" and not isinstance(self._runnable, ACLGraphWrapper):
            raise RuntimeError("vSpec MTP requires the Ascend MTP proposer to use a FULL ACL graph")
        return result

    load_model._vspec_mtp_strict_graph = True  # type: ignore[attr-defined]
    AscendSpecDecodeBaseProposer.load_model = load_model
    return True


def _sample_mtp_local_argmax(
    proposer: Any,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    """Select MTP draft tokens without gathering full-vocabulary logits."""
    get_top_tokens = getattr(proposer.model, "get_top_tokens", None)
    if get_top_tokens is None:
        raise RuntimeError(
            "MTP local argmax reduction requires the draft model to implement get_top_tokens()"
        )
    return get_top_tokens(hidden_states), None


def _patch_mtp_local_argmax_reduction() -> bool:
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )
    from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer

    applied = False
    original_compute_draft_token_ids = AscendSpecDecodeBaseProposer.compute_draft_token_ids
    if not getattr(
        original_compute_draft_token_ids,
        "_vspec_mtp_local_argmax",
        False,
    ):

        @wraps(original_compute_draft_token_ids)
        def compute_draft_token_ids(
            self: Any,
            hidden_states: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if self.method == "mtp" and self.use_local_argmax_reduction:
                if _GENERIC_MTP_LOCAL_ARGMAX.get():
                    calls = getattr(self, "_vspec_generic_mtp_local_argmax_calls", 0) + 1
                    self._vspec_generic_mtp_local_argmax_calls = calls
                    if calls in (1, 2, 3, 128, 1024):
                        logger.warning(
                            "vSpec generic MTP local argmax draft call=%d "
                            "(Python eager/capture; ACL graph replays bypass Python)",
                            calls,
                        )
                return _sample_mtp_local_argmax(self, hidden_states)
            return original_compute_draft_token_ids(
                self,
                hidden_states,
                *args,
                **kwargs,
            )

        compute_draft_token_ids._vspec_mtp_local_argmax = True  # type: ignore[attr-defined]
        AscendSpecDecodeBaseProposer.compute_draft_token_ids = compute_draft_token_ids
        applied = True

    original_step_sample = AscendStep3p5MTPProposer._sample_draft_tokens_for_step
    if not getattr(original_step_sample, "_vspec_mtp_local_argmax", False):

        @wraps(original_step_sample)
        def sample_draft_tokens_for_step(
            self: Any,
            hidden_states: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if self.method == "mtp" and self.use_local_argmax_reduction:
                return _sample_mtp_local_argmax(self, hidden_states)
            return original_step_sample(
                self,
                hidden_states,
                *args,
                **kwargs,
            )

        sample_draft_tokens_for_step._vspec_mtp_local_argmax = True  # type: ignore[attr-defined]
        AscendStep3p5MTPProposer._sample_draft_tokens_for_step = sample_draft_tokens_for_step
        applied = True

    # Ascend's generic _run_merged_draft bypasses compute_draft_token_ids()
    # unless enable_reduce_sample is set. Do not enable that global option:
    # it also changes target sampling and rejection. Scope only this branch
    # selection to the Qwen3.5 greedy MTP drafter, so the existing local
    # argmax implementation runs while the rest of Ascend retains its config.
    from vllm_ascend.spec_decode import llm_base_proposer as proposer_module

    original_run = AscendSpecDecodeBaseProposer._run_merged_draft
    if not getattr(original_run, "_vspec_generic_mtp_local_argmax", False):
        original_get_ascend_config = proposer_module.get_ascend_config

        class DraftOnlyAscendConfig:
            enable_reduce_sample = True

            def __init__(self, source: Any) -> None:
                self._source = source

            def __getattr__(self, name: str) -> Any:
                return getattr(self._source, name)

        @wraps(original_get_ascend_config)
        def draft_only_get_ascend_config() -> Any:
            config = original_get_ascend_config()
            if _GENERIC_MTP_LOCAL_ARGMAX.get():
                return DraftOnlyAscendConfig(config)
            return config

        @wraps(original_run)
        def run_merged_draft(self: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                self.method != "mtp"
                or not self.use_local_argmax_reduction
                or not _is_qwen35_mtp_config(self.vllm_config)
                or original_get_ascend_config().enable_reduce_sample
            ):
                return original_run(self, *args, **kwargs)
            if getattr(self, "_enable_probabilistic_draft_probs", False):
                raise RuntimeError("Generic MTP local argmax requires greedy draft selection")
            if getattr(self, "dynamic_spec", None) is not None:
                raise RuntimeError("Generic MTP local argmax does not support a logits-dependent selector")
            if proposer_module.lmhead_tp_enable():
                raise RuntimeError("Generic MTP local argmax has not validated fine-grained LM-head TP")
            if not hasattr(self.model, "get_top_tokens"):
                raise RuntimeError("Generic MTP local argmax requires get_top_tokens()")
            if not getattr(self, "_vspec_generic_mtp_local_argmax_logged", False):
                logger.warning(
                    "vSpec generic MTP draft-only local argmax branch active; "
                    "global Ascend reduce-sample remains disabled"
                )
                self._vspec_generic_mtp_local_argmax_logged = True
            token = _GENERIC_MTP_LOCAL_ARGMAX.set(True)
            try:
                return original_run(self, *args, **kwargs)
            finally:
                _GENERIC_MTP_LOCAL_ARGMAX.reset(token)

        run_merged_draft._vspec_generic_mtp_local_argmax = True  # type: ignore[attr-defined]
        AscendSpecDecodeBaseProposer._run_merged_draft = run_merged_draft
        proposer_module.get_ascend_config = draft_only_get_ascend_config
        applied = True
    return applied


def _patch_q35_eagle_mtp_metadata_fusion() -> bool:
    """Opt in only the actual Q35 Eagle-MTP continuation metadata path.

    Q35 uses AscendEagleProposer, not AscendStep3p5MTPProposer.  Define own
    methods on Eagle so other Ascend proposers retain the inherited methods.
    The actual kernel path additionally requires an exact Eagle instance,
    decode-only proposal, one attention group, and non-capture FULL/PIECEWISE.
    """
    if os.environ.get("VSPEC_DRAFT_Q35_METADATA_FUSION") != "1":
        return False

    from vllm.config import CUDAGraphMode
    from vllm_ascend.spec_decode import llm_base_proposer as ascend_base
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

    original_init = AscendEagleProposer.__init__
    original_propose = AscendEagleProposer._propose
    original_attn_update = AscendEagleProposer.attn_update_stack_num_spec_norm
    if getattr(original_propose, "_vspec_q35_eagle_metadata_fusion", False):
        return False

    trace = os.environ.get("VSPEC_DRAFT_Q35_METADATA_TRACE") == "1"

    @wraps(original_init)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        enabled = bool(
            type(self) is AscendEagleProposer
            and getattr(self, "method", None) == "mtp"
            and getattr(self, "uses_mrope", False)
            and _is_qwen35_mtp_config(self.vllm_config)
        )
        self._vspec_q35_metadata_fusion = enabled
        self._vspec_q35_metadata_runtime = False
        self._vspec_q35_metadata_logged = False
        self._vspec_q35_metadata_reject_logged = False
        self._vspec_q35_metadata_capture_logged = False
        self._vspec_q35_metadata_propose_seen: set[str] = set()
        if trace:
            print(
                "VSPEC_Q35_METADATA_EAGLE init: "
                f"enabled={int(enabled)} method={self.method} "
                f"uses_mrope={int(bool(self.uses_mrope))} "
                f"groups_at_init={len(self.draft_attn_groups)}",
                flush=True,
            )

    @wraps(original_propose)
    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Adaptive/runtime may have wrapped the inherited Ascend _propose as
        # a generic (*args, **kwargs) function before this hook is installed.
        # The production model runner supplies these fields by keyword, but
        # non-DCP runs report both counts as zero even for a mixed batch.
        # Require its independently computed uniform-decode descriptor too.
        # An unfamiliar/malformed caller retains the host path.
        prefill_known = type(kwargs.get("num_prefill_reqs")) is int
        num_prefill_reqs = kwargs.get("num_prefill_reqs")
        decode_count = kwargs.get("num_decode_reqs")
        decode_count_zero = type(decode_count) is int and decode_count == 0
        target_descriptor = kwargs.get("target_model_batch_desc")
        uniform_decode = getattr(target_descriptor, "uniform", None) is True
        previous_runtime = getattr(self, "_vspec_q35_metadata_runtime", False)
        self._vspec_q35_metadata_runtime = bool(
            type(self) is AscendEagleProposer
            and getattr(self, "_vspec_q35_metadata_fusion", False)
            and prefill_known
            and num_prefill_reqs == 0
            and decode_count_zero
            and uniform_decode
        )
        phase = (
            "decode" if num_prefill_reqs == 0 else "prefill"
        ) if prefill_known else "unknown"
        seen = getattr(self, "_vspec_q35_metadata_propose_seen", None)
        if trace and seen is not None and phase not in seen:
            seen.add(phase)
            print(
                "VSPEC_Q35_METADATA_EAGLE propose: "
                f"phase={phase} prefill_reqs={num_prefill_reqs if prefill_known else 'unknown'} "
                f"decode_reqs={decode_count if type(decode_count) is int else 'unknown'} "
                f"uniform={int(uniform_decode)} "
                f"runtime={int(bool(self._vspec_q35_metadata_runtime))}",
                flush=True,
            )
        try:
            return original_propose(self, *args, **kwargs)
        finally:
            self._vspec_q35_metadata_runtime = previous_runtime

    @wraps(original_attn_update)
    def attn_update_stack_num_spec_norm(
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
        runtime = bool(
            type(self) is AscendEagleProposer
            and getattr(self, "_vspec_q35_metadata_runtime", False)
        )
        # This update executes before Ascend installs a forward context, so
        # _EXTRA_CTX.capturing would assert.  The NPU stream query explicitly
        # returns False without initializing a missing NPU context.  Missing
        # or faulty capture-query APIs fail closed to the host updater.
        capturing = False
        if runtime:
            capture_checker = getattr(
                getattr(torch, "npu", None), "is_current_stream_capturing", None
            )
            if capture_checker is None:
                capturing = True
            else:
                try:
                    capturing = bool(capture_checker())
                except Exception:
                    capturing = True
        if (
            runtime
            and not capturing
            and ori_seq_len is None
            and ori_seq_len_cpu is None
            and slot_indices is None
            and mtp_slot_mapping is None
        ):
            from .draft_metadata_fusion import (
                can_fuse_q35_mtp_continuation,
                fuse_q35_mtp_continuation,
            )

            arguments = dict(
                draft_index=draft_index,
                old_common_metadata=old_common_metadata,
                batch_size=batch_size,
                input_batch_size=input_batch_size,
                used_update_positions=used_update_positions,
                attn_group=attn_group,
            )
            full_mode = aclgraph_runtime_mode == CUDAGraphMode.FULL
            piecewise_mode = aclgraph_runtime_mode == CUDAGraphMode.PIECEWISE
            if can_fuse_q35_mtp_continuation(
                self,
                arguments,
                full_mode=full_mode,
                piecewise_mode=piecewise_mode,
                capturing=False,
                spec_decode_state=ascend_base.AscendAttentionState.SpecDecoding,
            ):
                if trace and not self._vspec_q35_metadata_logged:
                    self._vspec_q35_metadata_logged = True
                    print(
                        "VSPEC_Q35_METADATA_FUSION hit: "
                        f"path=eagle_mtp mode={aclgraph_runtime_mode.name} "
                        f"batch={batch_size} input={input_batch_size} "
                        f"draft_index={draft_index}",
                        flush=True,
                    )
                return fuse_q35_mtp_continuation(
                    self,
                    arguments,
                    full_mode=full_mode,
                    attention_state=ascend_base.AscendAttentionState.SpecDecoding,
                    padding_slot_id=ascend_base.PADDING_SLOT_ID,
                )

        if trace and getattr(self, "_vspec_q35_metadata_fusion", False):
            log_attr = (
                "_vspec_q35_metadata_capture_logged"
                if capturing
                else "_vspec_q35_metadata_reject_logged"
            )
            if (runtime or capturing) and not getattr(self, log_attr, False):
                setattr(self, log_attr, True)
                state = getattr(old_common_metadata, "attn_state", None)
                block_table = getattr(old_common_metadata, "block_table_tensor", None)
                print(
                    "VSPEC_Q35_METADATA_FUSION fallback: "
                    f"path=eagle_mtp capturing={int(capturing)} "
                    f"runtime={int(runtime)} "
                    f"mode={getattr(aclgraph_runtime_mode, 'name', aclgraph_runtime_mode)} "
                    f"batch={batch_size} input={input_batch_size} "
                    f"draft_index={draft_index} groups={len(self.draft_attn_groups)} "
                    f"attn_state={getattr(state, 'name', type(state).__name__)} "
                    f"used_shape={tuple(used_update_positions.shape)} "
                    f"used_dtype={used_update_positions.dtype} "
                    f"block_shape={getattr(block_table, 'shape', None)}",
                    flush=True,
                )
        return original_attn_update(
            self,
            draft_index,
            old_common_metadata,
            batch_size,
            input_batch_size,
            used_update_positions,
            aclgraph_runtime_mode,
            ori_seq_len=ori_seq_len,
            ori_seq_len_cpu=ori_seq_len_cpu,
            slot_indices=slot_indices,
            mtp_slot_mapping=mtp_slot_mapping,
            attn_group=attn_group,
        )

    init._vspec_q35_eagle_metadata_fusion = True  # type: ignore[attr-defined]
    propose._vspec_q35_eagle_metadata_fusion = True  # type: ignore[attr-defined]
    attn_update_stack_num_spec_norm._vspec_q35_eagle_metadata_fusion = True  # type: ignore[attr-defined]
    AscendEagleProposer.__init__ = init
    AscendEagleProposer._propose = propose
    AscendEagleProposer.attn_update_stack_num_spec_norm = attn_update_stack_num_spec_norm
    logger.info("vSpec installed opt-in Q35 Eagle-MTP continuation metadata fusion")
    return True


def _uniform_step3p5_batch_size(
    common_attn_metadata: Any,
    num_speculative_tokens: int,
) -> int | None:
    """Return the real request count for a uniform Step3.5 verify window."""
    window_size = num_speculative_tokens + 1
    num_actual_tokens = int(common_attn_metadata.num_actual_tokens)
    if num_actual_tokens <= 0 or num_actual_tokens % window_size:
        return None

    batch_size = num_actual_tokens // window_size
    if batch_size <= 0 or batch_size > int(common_attn_metadata.num_reqs):
        return None

    query_start_loc_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
    if query_start_loc_cpu is None or query_start_loc_cpu.shape[0] < batch_size + 1:
        return None
    query_start_loc = query_start_loc_cpu[: batch_size + 1].tolist()
    if any(
        query_start_loc[index + 1] - query_start_loc[index] != window_size
        for index in range(batch_size)
    ):
        return None
    return batch_size


def _patch_step3p5_incremental_layers() -> bool:
    """Run Step3.5 layers after layer 0 with one token per request.

    The Ascend Step3.5 implementation currently rolls and reprocesses the full
    target verification window for every MTP layer. Current upstream vLLM only
    uses that window for layer 0; later layers consume the token drafted by the
    preceding layer. Rebuilding each later layer's attention metadata here also
    gives ACL graph capture the smaller, stable per-request shape.
    """
    if os.getenv("HUST_VSPEC_MTP_INCREMENTAL_STEPS", "1") == "0":
        return False

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer

    original_build = AscendStep3p5MTPProposer._build_step_attn_metadatas
    if getattr(original_build, "_vspec_incremental_step3p5", False):
        return False

    @wraps(original_build)
    def build_step_attn_metadatas(
        self: Any,
        common_attn_metadata: Any,
        *,
        graph_capture: bool = False,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        per_group, per_step = original_build(
            self,
            common_attn_metadata,
            graph_capture=graph_capture,
        )
        if self.method != "mtp" or self.num_speculative_tokens <= 1:
            return per_group, per_step

        batch_size = _uniform_step3p5_batch_size(
            common_attn_metadata,
            self.num_speculative_tokens,
        )
        if batch_size is None:
            return per_group, per_step
        if len(self.draft_attn_groups) < self.num_speculative_tokens:
            return per_group, per_step

        input_batch_size = int(common_attn_metadata.num_reqs) if self.use_cuda_graph else batch_size
        runtime_mode = CUDAGraphMode.FULL if self.use_cuda_graph else CUDAGraphMode.NONE
        token_indices = common_attn_metadata.query_start_loc[1 : batch_size + 1] - 1
        if self.uses_mrope:
            used_positions = self.mrope_positions[:, token_indices]
        else:
            used_positions = self.positions[token_indices]

        incremental_common = None
        for spec_step_idx in range(1, self.num_speculative_tokens):
            attn_group = self.draft_attn_groups[spec_step_idx]
            group_common = self._common_attn_metadata_for_group(
                common_attn_metadata,
                attn_group,
            )
            if incremental_common is not None:
                next_common = self.shallow_copy_metadata(incremental_common)
                next_common.block_table_tensor = group_common.block_table_tensor
                next_common.slot_mapping = group_common.slot_mapping
                incremental_common = next_common
            else:
                incremental_common = group_common

            incremental_common, attn_metadata = self.attn_update_stack_num_spec_norm(
                spec_step_idx,
                incremental_common,
                batch_size,
                input_batch_size,
                used_positions,
                runtime_mode,
                attn_group=attn_group,
            )
            per_group[spec_step_idx] = attn_metadata
            per_step[spec_step_idx] = {
                layer_name: attn_metadata for layer_name in attn_group.layer_names
            }
        return per_group, per_step

    def run_incremental_steps(
        self: Any,
        *,
        first_draft_token_ids: torch.Tensor,
        first_draft_probs: torch.Tensor | None,
        first_hidden_states: torch.Tensor,
        num_input_tokens: int,
        batch_size: int,
        token_indices_to_sample: torch.Tensor,
        target_positions: torch.Tensor,
        num_tokens: int,
        multi_steps_attn_metadata: list[dict[str, Any]],
        inputs_embeds: torch.Tensor | None,
        sampling_metadata: Any,
    ) -> torch.Tensor:
        del num_input_tokens, target_positions, num_tokens, inputs_embeds
        draft_probs_list = None if first_draft_probs is None else [first_draft_probs]
        draft_token_ids_list = [first_draft_token_ids]

        sample_indices = token_indices_to_sample[:batch_size]
        if self.uses_mrope:
            positions = self.mrope_positions[:, sample_indices]
        else:
            positions = self.positions[sample_indices]
        hidden_states = first_hidden_states[sample_indices]

        for draft_step in range(self.num_speculative_tokens - 1):
            spec_step_idx = draft_step + 1
            if spec_step_idx >= len(multi_steps_attn_metadata):
                raise AssertionError("Step3.5 MTP metadata must contain one entry per draft step")
            step_metadata = next(iter(multi_steps_attn_metadata[spec_step_idx].values()))
            input_batch_size = max(
                batch_size,
                int(step_metadata.query_start_loc.shape[0]) - 1,
            )
            _EXTRA_CTX.num_tokens = input_batch_size
            _EXTRA_CTX.num_accept_tokens = batch_size

            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0
                forward_context.attn_metadata = multi_steps_attn_metadata[spec_step_idx]

            input_ids = draft_token_ids_list[-1].int()
            positions = positions + 1
            if self.uses_mrope:
                exceeds_max_model_len = positions[0] >= self.vllm_config.model_config.max_model_len
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                exceeds_max_model_len = positions >= self.vllm_config.model_config.max_model_len
                clamped_positions = torch.where(
                    exceeds_max_model_len,
                    0,
                    positions,
                )

            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states.view(batch_size, -1)
            if input_batch_size > batch_size:
                self.input_ids[batch_size:input_batch_size].zero_()
                model_positions = self._get_positions(input_batch_size)
                if model_positions.ndim == 1:
                    model_positions[batch_size:input_batch_size].zero_()
                else:
                    model_positions[:, batch_size:input_batch_size].zero_()
                self.hidden_states[batch_size:input_batch_size].zero_()

            if self.supports_mm_inputs:
                self.inputs_embeds[:input_batch_size] = self.model.embed_input_ids(
                    self.input_ids[:input_batch_size]
                )
                model_input_ids = self.input_ids[:input_batch_size]
                step_inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                model_input_ids = self.input_ids[:input_batch_size]
                step_inputs_embeds = None

            model_positions = self._get_positions(input_batch_size)
            model_hidden_states = self.hidden_states[:input_batch_size]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(
                model_hidden_states,
                model_positions,
            )
            model_kwargs: dict[str, Any] = {
                "input_ids": model_input_ids,
                "positions": model_positions,
                "inputs_embeds": step_inputs_embeds,
                "spec_step_idx": spec_step_idx,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = model_hidden_states

            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
            last_hidden_states, model_positions, hidden_states = self.maybe_all_gather_and_unpad(
                last_hidden_states,
                model_positions,
                hidden_states,
            )

            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                last_hidden_states[:batch_size],
                sampling_metadata,
                spec_step_idx=spec_step_idx,
                num_indices=batch_size,
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            hidden_states = hidden_states[:batch_size]
            draft_token_ids_list.append(draft_token_ids)

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    build_step_attn_metadatas._vspec_incremental_step3p5 = True  # type: ignore[attr-defined]
    run_incremental_steps._vspec_incremental_step3p5 = True  # type: ignore[attr-defined]
    AscendStep3p5MTPProposer._build_step_attn_metadatas = build_step_attn_metadatas
    AscendStep3p5MTPProposer._run_window_draft_steps = run_incremental_steps
    logger.info(
        "vSpec enabled incremental Step3.5 MTP layers (full window for layer 0, "
        "one token per request for later layers)"
    )
    return True


def _remap_accepted_tokens_on_device(
    source: torch.Tensor,
    prev_positions: torch.Tensor,
    destination: torch.Tensor,
    safe_positions: torch.Tensor,
    new_request_mask: torch.Tensor,
    num_reqs: int,
) -> None:
    """Map previous-step accepted counts into the current batch order."""
    if num_reqs == 0:
        destination.fill_(1)
        return

    previous = prev_positions[:num_reqs]
    safe = safe_positions[:num_reqs]
    new_mask = new_request_mask[:num_reqs]
    torch.clamp(previous, min=0, out=safe)
    torch.lt(previous, 0, out=new_mask)
    torch.index_select(source, 0, safe, out=destination[:num_reqs])
    destination[:num_reqs].masked_fill_(new_mask, 1)
    destination[num_reqs:].fill_(1)


def _is_async_hybrid_mamba_runner(
    runner: Any,
    methods: frozenset[str],
) -> bool:
    speculative_config = getattr(runner, "speculative_config", None)
    return bool(
        getattr(speculative_config, "method", None) in methods
        and getattr(runner, "use_async_scheduling", False)
        and getattr(getattr(runner, "model_config", None), "is_hybrid", False)
        and getattr(getattr(runner, "cache_config", None), "mamba_cache_mode", None) == "align"
    )


def _is_async_mtp_mamba_runner(runner: Any) -> bool:
    return _is_async_hybrid_mamba_runner(runner, frozenset({"mtp"}))


def _synchronize_device_count_event(
    event: Any,
    runner: Any | None,
    original_synchronize: Any,
    *args: Any,
    diagnostic_host_wait: bool,
    validate_device_counts: bool,
    **kwargs: Any,
) -> Any:
    """Wait on accepted counts before any validation reads their CPU copy."""
    if runner is None or getattr(runner, "_vspec_mtp_device_count_phase", "idle") != "remap":
        return original_synchronize(event, *args, **kwargs)

    runner.prev_positions.copy_to_gpu(runner.input_batch.num_reqs)
    if diagnostic_host_wait or validate_device_counts:
        # CpuGpuBuffer.copy_to_cpu is nonblocking. Validation calls the
        # original CPU remap immediately after this event, so stream-only
        # ordering would leave its host source potentially stale.
        return original_synchronize(event, *args, **kwargs)
    torch.npu.current_stream().wait_event(event)
    return None


def _patch_mtp_cohort_refill() -> bool:
    """Select MTP replacement admission from measured online goodput."""
    from ..adaptive.refill import install_online_refill_patch

    return install_online_refill_patch(
        method="mtp",
        enabled_env="HUST_VSPEC_MTP_COHORT_REFILL",
        patch_marker="_vspec_mtp_cohort_refill",
    )


def _has_mamba_state_transition(
    scheduler_output: Any,
    mamba_state_idx: dict[str, int],
    input_batch: Any,
    requests: dict[str, Any],
    copy_bufs: Any,
) -> bool:
    """Return whether this step needs accepted-token-biased state copying."""
    reset_req_ids = set(scheduler_output.finished_req_ids)
    reset_req_ids.update(scheduler_output.preempted_req_ids or set())
    reset_req_ids.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
    block_size = copy_bufs.mamba_spec.block_size
    num_speculative_blocks = copy_bufs.mamba_spec.num_speculative_blocks

    for req_id in input_batch.req_ids:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if num_scheduled_tokens == 0:
            continue
        req_state = requests[req_id]
        prev_state_idx = None if req_id in reset_req_ids else mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_blocks = (
            req_state.num_computed_tokens + num_scheduled_tokens + block_size - 1
        ) // block_size + num_speculative_blocks
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            return True
    return False


def _patch_async_mtp_device_counts(
    *,
    methods: frozenset[str] = frozenset({"mtp"}),
    enabled: bool | None = None,
) -> bool:
    """Keep async hybrid-MTP acceptance state on the NPU between steps."""
    if enabled is None:
        enabled = os.getenv("HUST_VSPEC_MTP_DEVICE_COUNTS", "0") != "0"
    if not enabled:
        return False
    if not methods:
        raise ValueError("At least one speculative method is required")

    from vllm.v1.worker import mamba_utils
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_prepare_inputs = NPUModelRunner._prepare_inputs
    if getattr(original_prepare_inputs, "_vspec_mtp_device_counts", False):
        return False

    original_sync_counts = NPUModelRunner._sync_num_accepted_tokens
    original_update_states_after_model_execute = NPUModelRunner._update_states_after_model_execute
    original_event_synchronize = torch.npu.Event.synchronize
    original_preprocess_mamba = mamba_utils.preprocess_mamba
    diagnostic_host_wait = os.getenv("HUST_VSPEC_DEVICE_COUNTS_HOST_WAIT", "0") == "1"
    validate_device_counts = os.getenv("HUST_VSPEC_DEVICE_COUNTS_VALIDATE", "0") == "1"
    synchronize_device_remap = os.getenv("HUST_VSPEC_DEVICE_COUNTS_REMAP_SYNC", "0") == "1"
    publish_cpu_counts = os.getenv("HUST_VSPEC_DEVICE_COUNTS_PUBLISH_CPU", "0") == "1"
    trace_device_counts = os.getenv("HUST_VSPEC_DEVICE_COUNTS_TRACE", "0") == "1"
    fused_precopy_device_bias = os.getenv("HUST_VSPEC_DEVICE_COUNTS_FUSED_PRECOPY", "0") == "1"
    skip_postprocess_d2h = (
        os.getenv("HUST_VSPEC_DEVICE_COUNTS_SKIP_D2H", "0") == "1"
        or os.getenv("HUST_ISSUE18_DEVICE_COUNTS_SKIP_D2H", "0") == "1"
    )

    def materialize_pending_cpu_counts(runner: Any) -> None:
        """Publish a skipped postprocess count before a CPU-count path runs."""
        if not getattr(runner, "_vspec_mtp_device_count_pending_cpu", False):
            return
        ctx = runner._vspec_mtp_device_count_pending_ctx
        count = runner._vspec_mtp_device_count_pending_count
        event = runner.num_accepted_tokens_event
        if event is None:
            raise RuntimeError("A skipped accepted-count D2H requires a postprocess event")
        # The event is recorded after the fused postprocess. A host wait is
        # needed before publishing the CPU tensor, not merely a stream wait.
        original_event_synchronize(event)
        runner.num_accepted_tokens.cpu[:count].copy_(
            ctx.num_accepted_tokens_out[:count], non_blocking=True
        )
        torch.npu.current_stream().synchronize()
        runner._vspec_mtp_device_count_pending_cpu = False

    if fused_precopy_device_bias:
        precopy_owner = mamba_utils.MambaSpecDecodeGPUContext
        original_run_fused_precopy = precopy_owner.run_fused_precopy
        if not getattr(
            original_run_fused_precopy,
            _DEVICE_COUNT_PRECOPY_PATCH_MARKER,
            False,
        ):

            @wraps(original_run_fused_precopy)
            def run_fused_precopy(
                self: Any,
                num_reqs: int,
                state_idx_gpu: torch.Tensor,
                src_col_gpu: torch.Tensor,
                token_bias_gpu: torch.Tensor,
                idx_mapping: torch.Tensor | None,
            ) -> None:
                runner = getattr(self, "_vspec_mtp_device_count_runner", None)
                if (
                    runner is not None
                    and getattr(runner, "_vspec_mtp_device_count_phase", "idle") == "mapped"
                ):
                    torch.sub(
                        runner.num_accepted_tokens.gpu[:num_reqs],
                        1,
                        out=token_bias_gpu[:num_reqs],
                    )
                return original_run_fused_precopy(
                    self,
                    num_reqs,
                    state_idx_gpu,
                    src_col_gpu,
                    token_bias_gpu,
                    idx_mapping,
                )

            setattr(
                run_fused_precopy,
                _DEVICE_COUNT_PRECOPY_PATCH_MARKER,
                True,
            )
            precopy_owner.run_fused_precopy = run_fused_precopy

    def install_buffer_hook(runner: Any) -> None:
        accepted = runner.num_accepted_tokens
        if getattr(accepted, "_vspec_mtp_device_counts", False):
            return

        original_copy_to_gpu = accepted.copy_to_gpu

        def copy_to_gpu(n: int | None = None) -> torch.Tensor:
            phase = getattr(runner, "_vspec_mtp_device_count_phase", "idle")
            if phase == "remap":
                ctx = runner._get_mamba_bufs().postprocess_align
                assert ctx is not None
                num_reqs = runner.input_batch.num_reqs
                runner.prev_positions.copy_to_gpu(num_reqs)
                _remap_accepted_tokens_on_device(
                    ctx.num_accepted_tokens_out,
                    runner.prev_positions.gpu,
                    accepted.gpu,
                    runner._vspec_mtp_safe_prev_positions,
                    runner._vspec_mtp_new_request_mask,
                    num_reqs,
                )
                if synchronize_device_remap:
                    torch.npu.current_stream().synchronize()
                runner._vspec_mtp_device_count_phase = "mapped"
                return accepted.gpu if n is None else accepted.gpu[:n]
            if phase == "preprocessed":
                ctx = runner._get_mamba_bufs().postprocess_align
                assert ctx is not None
                num_reqs = runner.input_batch.num_reqs
                _remap_accepted_tokens_on_device(
                    ctx.num_accepted_tokens_out,
                    runner.prev_positions.gpu,
                    accepted.gpu,
                    runner._vspec_mtp_safe_prev_positions,
                    runner._vspec_mtp_new_request_mask,
                    num_reqs,
                )
                runner._vspec_mtp_device_count_phase = "idle"
                runner._vspec_mtp_device_count_fast_step = True
                return accepted.gpu if n is None else accepted.gpu[:n]
            return original_copy_to_gpu(n)

        accepted.copy_to_gpu = copy_to_gpu
        accepted._vspec_mtp_device_counts = True

    def install_postprocess_scratch_hook(runner: Any, mamba_bufs: Any) -> None:
        mamba_bufs.preprocess._vspec_mtp_runner = runner
        if publish_cpu_counts or validate_device_counts or diagnostic_host_wait:
            return
        original_postprocess_mamba_align_gpu = mamba_utils.postprocess_mamba_align_gpu
        if getattr(original_postprocess_mamba_align_gpu, "__name__", None) == "_postprocess_mamba_align_gpu_cpu_fallback":
            return
        # This opt-in bypasses only the known upstream function's final D2H.
        # Other wrappers may have additional work that cannot be skipped.
        supports_device_only_postprocess = (
            original_postprocess_mamba_align_gpu.__module__ == mamba_utils.__name__
            and original_postprocess_mamba_align_gpu.__name__ == "postprocess_mamba_align_gpu"
        )
        if getattr(original_postprocess_mamba_align_gpu, "_vspec_mtp_d2h_scratch", False):
            return
        if skip_postprocess_d2h and not supports_device_only_postprocess:
            logger.warning(
                "vSpec accepted-count D2H skip disabled for unsupported postprocess %s.%s",
                original_postprocess_mamba_align_gpu.__module__,
                original_postprocess_mamba_align_gpu.__name__,
            )

        @wraps(original_postprocess_mamba_align_gpu)
        def postprocess_mamba_align_gpu(*post_args: Any, **post_kwargs: Any) -> Any:
            bufs = post_kwargs.get("bufs")
            post_runner = getattr(getattr(bufs, "preprocess", None), "_vspec_mtp_runner", None)
            ctx = getattr(bufs, "postprocess_align", None)
            use_scratch = post_runner is not None and ctx is not None and ctx.is_initialized
            if post_runner is not None:
                post_runner._vspec_mtp_device_count_last_d2h_scratch = use_scratch
            use_device_only = bool(
                skip_postprocess_d2h
                and supports_device_only_postprocess
                and use_scratch
                and getattr(post_runner, "_vspec_mtp_device_count_fast_step", False)
                and getattr(post_runner, "_vspec_mtp_device_count_phase", None) == "idle"
                and post_runner.num_accepted_tokens_event is not None
                and _is_async_hybrid_mamba_runner(post_runner, methods)
            )
            if use_device_only:
                # The original routine's only work after this fused call is
                # D2H of num_accepted_tokens_out. The next fast step consumes
                # that GPU tensor directly; defer D2H until a CPU path needs it.
                ctx.run_fused_postprocess(
                    num_reqs=post_kwargs["num_reqs"],
                    num_accepted_tokens_gpu=post_kwargs["num_accepted_tokens_gpu"],
                    mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
                    num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,
                    num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,
                    num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,
                )
                post_runner._vspec_mtp_device_count_pending_ctx = ctx
                post_runner._vspec_mtp_device_count_pending_count = post_kwargs["num_reqs"]
                post_runner._vspec_mtp_device_count_pending_cpu = True
                post_runner._vspec_mtp_device_count_last_d2h_scratch = False
                post_runner._vspec_mtp_device_count_fast_step = False
                skipped = getattr(post_runner, "_vspec_mtp_device_count_skipped_d2h_steps", 0) + 1
                post_runner._vspec_mtp_device_count_skipped_d2h_steps = skipped
                if skipped == 1:
                    logger.warning(
                        "vSpec accepted-count postprocess D2H skipped after fused GPU kernel"
                    )
                return None
            if post_runner is not None:
                materialize_pending_cpu_counts(post_runner)
                post_runner._vspec_mtp_device_count_fast_step = False
            if use_scratch:
                scratch = getattr(post_runner, "_vspec_mtp_device_count_d2h_scratch", None)
                if scratch is None:
                    scratch = torch.empty_like(
                        post_runner.num_accepted_tokens.cpu,
                        pin_memory=True,
                    )
                    post_runner._vspec_mtp_device_count_d2h_scratch = scratch
                post_runner._vspec_mtp_device_count_d2h_count = post_kwargs["num_reqs"]
                post_kwargs["num_accepted_tokens_cpu_tensor"] = scratch
            return original_postprocess_mamba_align_gpu(*post_args, **post_kwargs)

        postprocess_mamba_align_gpu._vspec_mtp_d2h_scratch = True  # type: ignore[attr-defined]
        mamba_utils.postprocess_mamba_align_gpu = postprocess_mamba_align_gpu

    @wraps(original_prepare_inputs)
    def prepare_inputs(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal original_preprocess_mamba
        if not _is_async_hybrid_mamba_runner(self, methods):
            self._vspec_mtp_device_count_fast_step = False
            materialize_pending_cpu_counts(self)
            return original_prepare_inputs(self, *args, **kwargs)

        self._vspec_mtp_device_count_fast_step = False

        if mamba_utils.preprocess_mamba is not preprocess_mamba:
            # Ascend installs its worker Mamba patch after plugin registration.
            # Restore this wrapper around the Ascend implementation, not the
            # upstream function captured before worker initialization.
            original_preprocess_mamba = mamba_utils.preprocess_mamba
            mamba_utils.preprocess_mamba = preprocess_mamba
            logger.warning(
                "vSpec restored accepted-count Mamba wrapper around %s.%s",
                original_preprocess_mamba.__module__,
                original_preprocess_mamba.__name__,
            )

        mamba_bufs = self._get_mamba_bufs()
        install_postprocess_scratch_hook(self, mamba_bufs)
        ctx = mamba_bufs.postprocess_align
        if ctx is None or not ctx.is_initialized:
            self._vspec_mtp_device_count_phase = "fallback"
            materialize_pending_cpu_counts(self)
            return original_prepare_inputs(self, *args, **kwargs)
        if fused_precopy_device_bias:
            ctx._vspec_mtp_device_count_runner = self

        install_buffer_hook(self)
        if not hasattr(self, "_vspec_mtp_safe_prev_positions"):
            self._vspec_mtp_safe_prev_positions = torch.empty_like(self.prev_positions.gpu)
            self._vspec_mtp_new_request_mask = torch.empty_like(
                self.prev_positions.gpu,
                dtype=torch.bool,
            )
            if validate_device_counts:
                self._vspec_mtp_validated_counts = torch.empty_like(self.num_accepted_tokens.gpu)
                self._vspec_mtp_validation_steps = 0

        self._vspec_mtp_device_count_phase = "remap"
        if self.num_accepted_tokens_event is not None:
            _DEVICE_COUNT_EVENT_OWNERS[id(self.num_accepted_tokens_event)] = self
        try:
            return original_prepare_inputs(self, *args, **kwargs)
        except Exception:
            self._vspec_mtp_device_count_phase = "idle"
            self._vspec_mtp_device_count_fast_step = False
            raise

    @wraps(original_sync_counts)
    def sync_num_accepted_tokens(
        self: Any,
        num_reqs: int,
        has_prev_mapping: bool,
    ) -> None:
        if getattr(self, "_vspec_mtp_device_count_phase", "idle") != "remap":
            materialize_pending_cpu_counts(self)
            scratch = getattr(self, "_vspec_mtp_device_count_d2h_scratch", None)
            if scratch is not None and getattr(
                self, "_vspec_mtp_device_count_last_d2h_scratch", False
            ):
                if self.num_accepted_tokens_event is not None:
                    original_event_synchronize(self.num_accepted_tokens_event)
                count = self._vspec_mtp_device_count_d2h_count
                self.num_accepted_tokens.cpu[:count].copy_(scratch[:count])
            original_sync_counts(self, num_reqs, has_prev_mapping)
            return
        if publish_cpu_counts:
            original_sync_counts(self, num_reqs, has_prev_mapping)
            return
        if not validate_device_counts:
            return

        host_source = self.num_accepted_tokens.np.copy()
        original_sync_counts(self, num_reqs, has_prev_mapping)
        ctx = self._get_mamba_bufs().postprocess_align
        assert ctx is not None
        validated = self._vspec_mtp_validated_counts
        _remap_accepted_tokens_on_device(
            ctx.num_accepted_tokens_out,
            self.prev_positions.gpu,
            validated,
            self._vspec_mtp_safe_prev_positions,
            self._vspec_mtp_new_request_mask,
            num_reqs,
        )
        torch.npu.current_stream().synchronize()
        expected = self.num_accepted_tokens.np[:num_reqs].tolist()
        actual = validated[:num_reqs].cpu().tolist()
        self._vspec_mtp_validation_steps += 1
        if expected != actual or self._vspec_mtp_validation_steps <= 3:
            source_len = num_reqs
            if has_prev_mapping and num_reqs:
                source_len = max(
                    source_len,
                    int(self.prev_positions.np[:num_reqs].max(initial=-1)) + 1,
                )
            from vllm.logger import logger as vllm_logger

            vllm_logger.warning(
                "vSpec accepted-count validation step=%d match=%s "
                "prev_host=%s prev_device=%s source_host=%s source_device=%s "
                "expected=%s actual=%s",
                self._vspec_mtp_validation_steps,
                expected == actual,
                self.prev_positions.np[:num_reqs].tolist(),
                self.prev_positions.gpu[:num_reqs].cpu().tolist(),
                host_source[:source_len].tolist(),
                ctx.num_accepted_tokens_out[:source_len].cpu().tolist(),
                expected,
                actual,
            )
        self._vspec_mtp_device_count_phase = "fallback"

    @wraps(original_update_states_after_model_execute)
    def update_states_after_model_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_async_hybrid_mamba_runner(self, methods):
            mamba_bufs = self._get_mamba_bufs()
            install_postprocess_scratch_hook(self, mamba_bufs)
        return original_update_states_after_model_execute(self, *args, **kwargs)

    @wraps(original_event_synchronize)
    def event_synchronize(self: Any, *args: Any, **kwargs: Any) -> Any:
        runner = _DEVICE_COUNT_EVENT_OWNERS.get(id(self))
        return _synchronize_device_count_event(
            self,
            runner,
            original_event_synchronize,
            *args,
            diagnostic_host_wait=diagnostic_host_wait or publish_cpu_counts,
            validate_device_counts=validate_device_counts,
            **kwargs,
        )

    @wraps(original_preprocess_mamba)
    def preprocess_mamba(*args: Any, **kwargs: Any) -> Any:
        copy_bufs = kwargs.get("copy_bufs")
        if copy_bufs is None and len(args) >= 9:
            copy_bufs = args[8]
        runner = getattr(copy_bufs, "_vspec_mtp_runner", None)
        if runner is None or getattr(runner, "_vspec_mtp_device_count_phase", "idle") != "mapped":
            return original_preprocess_mamba(*args, **kwargs)

        scheduler_output = args[0]
        mamba_state_idx = args[3]
        input_batch = args[4]
        requests = args[5]
        has_state_transition = _has_mamba_state_transition(
            scheduler_output,
            mamba_state_idx,
            input_batch,
            requests,
            copy_bufs,
        )
        if trace_device_counts:
            trace_step = getattr(runner, "_vspec_mtp_device_trace_step", 0) + 1
            runner._vspec_mtp_device_trace_step = trace_step
            if trace_step <= 3:
                from vllm.logger import logger as vllm_logger

                vllm_logger.warning(
                    "vSpec accepted-count device trace step=%d requests=%d "
                    "state_transition=%s phase=%s",
                    trace_step,
                    len(input_batch.req_ids),
                    has_state_transition,
                    runner._vspec_mtp_device_count_phase,
                )
        device_precopy_available = bool(
            fused_precopy_device_bias
            and getattr(copy_bufs, "_vspec_eagle3_align_ctx", None) is not None
        )
        if has_state_transition and not device_precopy_available:
            num_reqs = len(input_batch.req_ids)
            ctx = runner._get_mamba_bufs().postprocess_align
            assert ctx is not None
            _remap_accepted_tokens_on_device(
                ctx.num_accepted_tokens_out,
                runner.prev_positions.gpu,
                runner.num_accepted_tokens.gpu,
                runner._vspec_mtp_safe_prev_positions,
                runner._vspec_mtp_new_request_mask,
                num_reqs,
            )
            runner.num_accepted_tokens.copy_to_cpu(num_reqs)
            torch.npu.current_stream().synchronize()
            input_batch.num_accepted_tokens_cpu[:num_reqs] = runner.num_accepted_tokens.np[
                :num_reqs
            ]
            runner._vspec_mtp_device_count_pending_cpu = False
            runner._vspec_mtp_device_count_phase = "fallback"
            return original_preprocess_mamba(*args, **kwargs)

        result = original_preprocess_mamba(*args, **kwargs)
        runner._vspec_mtp_device_count_phase = "preprocessed"
        return result

    prepare_inputs._vspec_mtp_device_counts = True  # type: ignore[attr-defined]
    sync_num_accepted_tokens._vspec_mtp_device_counts = True  # type: ignore[attr-defined]
    update_states_after_model_execute._vspec_mtp_device_counts = True  # type: ignore[attr-defined]
    event_synchronize._vspec_mtp_device_counts = True  # type: ignore[attr-defined]
    preprocess_mamba._vspec_mtp_device_counts = True  # type: ignore[attr-defined]
    NPUModelRunner._prepare_inputs = prepare_inputs
    NPUModelRunner._sync_num_accepted_tokens = sync_num_accepted_tokens
    NPUModelRunner._update_states_after_model_execute = update_states_after_model_execute
    torch.npu.Event.synchronize = event_synchronize
    mamba_utils.preprocess_mamba = preprocess_mamba
    from vllm.logger import logger as vllm_logger

    vllm_logger.info(
        "vSpec enabled device-resident accepted-token remapping for %s between "
        "Mamba block transitions (host_wait=%s, remap_sync=%s, "
        "publish_cpu=%s, fused_precopy_bias=%s, skip_d2h=%s, validate=%s)",
        ", ".join(sorted(methods)),
        diagnostic_host_wait,
        synchronize_device_remap,
        publish_cpu_counts,
        fused_precopy_device_bias,
        skip_postprocess_d2h,
        validate_device_counts,
    )
    return True


def _is_qwen35_mtp_config(vllm_config: Any) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if getattr(speculative_config, "method", None) != "mtp":
        return False
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_types = {
        getattr(hf_config, "model_type", ""),
        getattr(hf_text_config, "model_type", ""),
    }
    return any(str(model_type).startswith("qwen3_5") for model_type in model_types)


def _annotate_qwen35_mtp_kv_groups(
    vllm_config: Any,
    kv_cache_spec: dict[str, Any],
    kv_cache_groups: list[Any],
) -> tuple[int, ...]:
    """Identify Qwen3.5 MTP attention groups without flagging Mamba groups."""
    if not _is_qwen35_mtp_config(vllm_config):
        return ()

    draft_layer_names = {
        name for name in kv_cache_spec if name.startswith("mtp.") or ".mtp." in name
    }
    if not draft_layer_names:
        raise RuntimeError(
            "vSpec could not identify the Qwen3.5 MTP KV-cache layer; refusing "
            "the upstream all-group fallback because it disables hybrid APC reuse"
        )

    annotated: list[int] = []
    for group_id, group in enumerate(kv_cache_groups):
        if draft_layer_names.intersection(group.layer_names):
            group.is_eagle_group = True
            annotated.append(group_id)
    if not annotated:
        raise RuntimeError(
            "vSpec found Qwen3.5 MTP layers but none belonged to a KV-cache group: "
            + ", ".join(sorted(draft_layer_names))
        )
    return tuple(annotated)


def _patch_mtp_hybrid_kv_group_annotation() -> bool:
    """Keep hybrid Mamba groups eligible for prefix-cache reuse with MTP."""
    from vllm.v1.core import kv_cache_utils

    original_annotate = kv_cache_utils._annotate_eagle_groups
    if getattr(original_annotate, "_vspec_qwen35_mtp_hybrid", False):
        return False

    @wraps(original_annotate)
    def annotate_eagle_groups(
        vllm_config: Any,
        kv_cache_spec: dict[str, Any],
        kv_cache_groups: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        original_annotate(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            *args,
            **kwargs,
        )
        group_ids = _annotate_qwen35_mtp_kv_groups(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
        )
        if group_ids:
            logger.info(
                "vSpec marked Qwen3.5 MTP KV-cache group(s) %s; hybrid Mamba "
                "groups remain eligible for APC reuse",
                group_ids,
            )

    annotate_eagle_groups._vspec_qwen35_mtp_hybrid = True  # type: ignore[attr-defined]
    kv_cache_utils._annotate_eagle_groups = annotate_eagle_groups
    return True


def _is_uniform_decode_fallback(
    dispatcher: Any,
    uniform_decode: bool,
    runtime_mode: Any,
) -> bool:
    from vllm.config import CUDAGraphMode

    return bool(
        dispatcher.keys_initialized and uniform_decode and runtime_mode != CUDAGraphMode.FULL
    )


def _patch_mtp_sync_trace() -> bool:
    event_type = torch.npu.Event
    original_synchronize = event_type.synchronize
    if getattr(original_synchronize, "_vspec_mtp_sync_trace", False):
        return False

    @wraps(original_synchronize)
    def synchronize(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = original_synchronize(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms >= 2:
            logger.warning(
                "vSpec MTP event synchronization took %.2f ms: %s",
                elapsed_ms,
                "".join(traceback.format_stack(limit=6)),
            )
        return result

    synchronize._vspec_mtp_sync_trace = True  # type: ignore[attr-defined]
    event_type.synchronize = synchronize
    return True


def _multistep_install_requested() -> bool:
    # VALIDATE without the base opt-in must reach the installer and hard-fail.
    return (
        os.getenv("HUST_ISSUE18_Q35_EAGLE_MULTISTEP", "0") == "1"
        or os.getenv("HUST_ISSUE18_Q35_EAGLE_MULTISTEP_VALIDATE", "0") == "1"
    )


def apply_mtp_patches(settings: PluginSettings) -> bool:
    """Use the host's native MTP implementation after ABI validation.

    Qwen3.5 stores its MTP head in the target checkpoint. vSpec therefore
    contributes launch policy and compatibility checks without replacing the
    host model or proposer implementation.
    """
    _validate_ascend_custom_ops()
    applied = _patch_mtp_hybrid_kv_group_annotation()
    if os.getenv("HUST_ISSUE18_Q35_DRAFT_QSL_DEVICE", "0") == "1":
        # Wrap the inherited base proposal first.  The Eagle-only metadata
        # wrapper installed below then captures this scoped QSL wrapper and
        # preserves its ContextVar lifetime through the original proposal.
        from .mtp_qsl_device import apply_q35_draft_qsl_device_patch

        applied = apply_q35_draft_qsl_device_patch() or applied
    if os.getenv("HUST_VSPEC_MTP_SYNC_TRACE") == "1":
        applied = _patch_mtp_sync_trace() or applied
    applied = _patch_mtp_mamba_group_api() or applied
    applied = _patch_mtp_local_argmax_reduction() or applied
    if os.getenv('HUST_ISSUE18_Q35_COMPACT_BODY', '0') == '1':
        from .mtp_compact_body import install_compact_mtp_body

        applied = install_compact_mtp_body() or applied
    applied = _patch_q35_eagle_mtp_metadata_fusion() or applied
    if _multistep_install_requested():
        # Private, default-off prototype.  It must wrap the Eagle per-step
        # fusion so an ineligible proposal can still use that vetted path.
        from .eagle_multistep_metadata import apply_q35_eagle_multistep_metadata_patch

        applied = apply_q35_eagle_multistep_metadata_patch() or applied
    applied = _patch_step3p5_incremental_layers() or applied
    applied = _patch_async_mtp_device_counts() or applied
    if os.getenv("HUST_ISSUE18_Q35_GDN_MTP_SHARE", "0") == "1":
        from .gdn_mtp_share import apply_q35_mtp_gdn_share_patch

        applied = apply_q35_mtp_gdn_share_patch() or applied
    if os.getenv("HUST_ISSUE18_Q35_EARLY_VALID_COUNT", "0") == "1":
        from .mtp_early_valid_count import apply_q35_early_valid_count_patch

        applied = apply_q35_early_valid_count_patch() or applied
    applied = _patch_mtp_cohort_refill() or applied
    if os.getenv('HUST_ISSUE18_TARGET_GREEDY_LOCALREDUCE') == '1':
        from .target_localreduce import install_target_localreduce

        applied = install_target_localreduce() or applied
    if not settings.mtp_strict_graph:
        return applied

    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    applied = _patch_mtp_drafter_graph_guard() or applied
    if os.getenv('HUST_ISSUE18_Q35_MTP_DRAFT_ACTIVE_SIZE', '0') != '0':
        from .mtp_active_head import install_q35_active_proposal_head

        applied = install_q35_active_proposal_head() or applied
    original_dispatch = CudagraphDispatcher.dispatch
    if getattr(original_dispatch, "_vspec_mtp_strict_graph", False):
        return applied

    trace_graph = os.getenv("HUST_VSPEC_MTP_GRAPH_TRACE") == "1"
    graph_counts: Counter[tuple[int, bool, str, int]] = Counter()

    @wraps(original_dispatch)
    def dispatch(
        self: Any,
        num_tokens: int,
        uniform_decode: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        runtime_mode, descriptor = original_dispatch(
            self,
            num_tokens,
            uniform_decode,
            *args,
            **kwargs,
        )
        if trace_graph:
            graph_counts[
                (num_tokens, uniform_decode, runtime_mode.name, descriptor.num_tokens)
            ] += 1
            if sum(graph_counts.values()) % 128 == 0:
                logger.warning("vSpec MTP graph dispatch counts: %s", graph_counts.most_common(12))
        if _is_uniform_decode_fallback(self, uniform_decode, runtime_mode):
            raise RuntimeError(
                "vSpec MTP refused an eager decode fallback: "
                f"num_tokens={num_tokens}, descriptor={descriptor}, "
                f"configured_mode={self.cudagraph_mode}"
            )
        return runtime_mode, descriptor

    dispatch._vspec_mtp_strict_graph = True  # type: ignore[attr-defined]
    CudagraphDispatcher.dispatch = dispatch
    return True
