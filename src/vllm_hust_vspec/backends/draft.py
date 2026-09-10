"""Ascend runtime patches used by the optimized serial Draft backend."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import vllm_ascend.ops.register_custom_ops  # noqa: F401
import vllm_ascend.spec_decode.llm_base_proposer as ascend_base_proposer
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from ..config import PluginSettings
from ..host_compat import call_with_supported_kwargs
from .draft_vocab import _configure_serial_draft_active_vocab

PATCH_MARKER = "_vllm_hust_vspec_draft_patched"


def _install_inner_logits_processor_alias(model: Any) -> bool:
    """Bridge the current Ascend reduce-sample lookup for causal LMs."""
    inner_model = getattr(model, "model", None)
    logits_processor = getattr(model, "logits_processor", None)
    if inner_model is None or logits_processor is None or hasattr(inner_model, "logits_processor"):
        return False

    # Avoid registering the same nn.Module under a second parent path. The
    # Ascend proposer only needs the processor's _gather_logits method.
    object.__setattr__(inner_model, "logits_processor", logits_processor)
    return True


def _call_host_query_padding(
    host_method: Any,
    runner: Any,
    query_start_loc: Any,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_reqs: int,
    cudagraph_runtime_mode: Any,
    batch_desc_num_reqs: int | None,
    batch_desc_uniform: bool | None,
) -> int:
    kwargs = {
        "cudagraph_runtime_mode": cudagraph_runtime_mode,
        "batch_desc_num_reqs": batch_desc_num_reqs,
    }
    if "batch_desc_uniform" in inspect.signature(host_method).parameters:
        kwargs["batch_desc_uniform"] = batch_desc_uniform
    return host_method(
        runner,
        query_start_loc,
        num_tokens_padded,
        num_reqs_padded,
        num_reqs,
        **kwargs,
    )


def _bind_host_attn_update(
    host_method: Any,
    proposer: Any,
    draft_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> inspect.BoundArguments:
    return inspect.signature(host_method).bind(
        proposer,
        draft_index,
        *args,
        **kwargs,
    )


class CompactPiecewiseDraftModel:
    """Run serial continuation steps at request-batch granularity."""

    def __init__(self, proposer: Any, model: Any) -> None:
        self.proposer = proposer
        self.model_instance = model
        self.num_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model_instance, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.num_calls += 1
        if self.num_calls == 1:
            return self.model_instance(*args, **kwargs)

        num_tokens = kwargs["input_ids"].shape[0]
        runtime_mode, batch_descriptor = self.proposer.runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens,
            uniform_decode=False,
            has_lora=False,
        )
        if runtime_mode != CUDAGraphMode.PIECEWISE:
            return self.model_instance(*args, **kwargs)

        num_tokens_padded = batch_descriptor.num_tokens
        kwargs["input_ids"] = self.proposer.input_ids[:num_tokens_padded]
        kwargs["positions"] = self.proposer._get_positions(num_tokens_padded)
        if kwargs.get("hidden_states") is not None:
            kwargs["hidden_states"] = self.proposer.hidden_states[:num_tokens_padded]
        if kwargs.get("inputs_embeds") is not None:
            kwargs["inputs_embeds"] = self.proposer.inputs_embeds[:num_tokens_padded]

        forward_context = get_forward_context()
        original_mode = forward_context.cudagraph_runtime_mode
        original_descriptor = forward_context.batch_descriptor
        original_num_tokens = forward_context.num_tokens
        original_padded_num_tokens = forward_context.padded_num_tokens
        original_extra_num_tokens = _EXTRA_CTX.num_tokens
        forward_context.cudagraph_runtime_mode = runtime_mode
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens_padded,
            uniform=False,
        )
        forward_context.num_tokens = num_tokens_padded
        forward_context.padded_num_tokens = num_tokens_padded
        _EXTRA_CTX.num_tokens = num_tokens_padded
        try:
            return self.model_instance(*args, **kwargs)
        finally:
            forward_context.cudagraph_runtime_mode = original_mode
            forward_context.batch_descriptor = original_descriptor
            forward_context.num_tokens = original_num_tokens
            forward_context.padded_num_tokens = original_padded_num_tokens
            _EXTRA_CTX.num_tokens = original_extra_num_tokens


def apply_draft_patches(settings: PluginSettings) -> bool:
    """Install process-local monkey patches once and return whether applied."""
    applied = False
    if settings.draft_target_active_vocab:
        from .eagle_target import apply_target_active_vocab_patch

        applied = apply_target_active_vocab_patch(
            active_ids_environment=("VLLM_ASCEND_DRAFT_TARGET_ACTIVE_VOCAB_IDS_PATH"),
            required_method="draft_model",
            feature_name="Draft Target",
        )
    if getattr(AscendDraftModelProposer, PATCH_MARKER, False):
        return applied

    original_ascend_init = AscendDraftModelProposer.__init__
    original_draft_dummy_run = ascend_base_proposer.AscendSpecDecodeBaseProposer.dummy_run
    original_pad_query_start_loc = NPUModelRunner._pad_query_start_loc_for_fia
    original_run_merged_draft = ascend_base_proposer.AscendSpecDecodeBaseProposer._run_merged_draft
    original_attn_update = (
        ascend_base_proposer.AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm
    )
    original_draft_propose = ascend_base_proposer.AscendSpecDecodeBaseProposer._propose
    original_compute_slot_mapping = ascend_base_proposer.compute_new_slot_mapping

    from .eagle_rejection import apply_linear_rejection_patch

    apply_linear_rejection_patch()

    def init_draft_proposer(
        self: Any,
        vllm_config: Any,
        device: Any,
        runner: Any = None,
    ) -> None:
        if settings.assume_shared_tokenizer:
            DraftModelProposer.__init__(self, vllm_config, device, runner)
            # The known Qwen2.5 pair shares token IDs and differs only in LM-head
            # padding, so runtime remapping is unnecessary.
            self.use_heterogeneous_vocab = False
            self.vocab_mapping = None
        else:
            original_ascend_init(self, vllm_config, device, runner)

        # The proposer is constructed before runner._use_aclgraph() is finalized.
        self.use_cuda_graph = (
            not vllm_config.model_config.enforce_eager and not self.speculative_config.enforce_eager
        )

    def draft_dummy_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        _install_inner_logits_processor_alias(self.model)
        if settings.draft_active_vocab:
            _configure_serial_draft_active_vocab(self)
        if self.use_cuda_graph and not isinstance(self._runnable, ACLGraphWrapper):
            self.update_stream = torch.npu.Stream()
            self._runnable = call_with_supported_kwargs(
                ACLGraphWrapper,
                self._run_merged_draft,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=(
                    self.use_eagle or settings.use_merged_full or settings.merged_full_max_batch > 0
                ),
                enable_enpu=self.enable_enpu,
                is_draft_model=True,
            )
        return original_draft_dummy_run(self, *args, **kwargs)

    def pad_query_start_loc_for_fia(
        self: Any,
        query_start_loc: Any,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: Any = None,
        batch_desc_num_reqs: int | None = None,
        batch_desc_uniform: bool | None = None,
    ) -> int:
        if (
            getattr(self, "_draft_merged_full_padding", False)
            and cudagraph_runtime_mode == CUDAGraphMode.FULL
            and batch_desc_num_reqs is not None
        ):
            if num_reqs > batch_desc_num_reqs:
                raise RuntimeError(
                    "draft graph descriptor has fewer request rows than the "
                    f"runtime batch: actual={num_reqs}, padded={batch_desc_num_reqs}"
                )

            last_loc = int(query_start_loc.np[num_reqs])
            remaining_tokens = num_tokens_padded - last_loc
            remaining_reqs = batch_desc_num_reqs - num_reqs
            if remaining_tokens < 0:
                raise RuntimeError(
                    "draft graph has fewer token rows than the runtime batch: "
                    f"actual={last_loc}, padded={num_tokens_padded}"
                )

            if remaining_reqs:
                base, extra = divmod(remaining_tokens, remaining_reqs)
                cursor = last_loc
                for offset in range(remaining_reqs):
                    cursor += base + (offset < extra)
                    query_start_loc.np[num_reqs + offset + 1] = cursor
            elif remaining_tokens:
                query_start_loc.np[num_reqs] = num_tokens_padded

            query_start_loc.copy_to_gpu()
            return batch_desc_num_reqs

        return _call_host_query_padding(
            original_pad_query_start_loc,
            self,
            query_start_loc,
            num_tokens_padded,
            num_reqs_padded,
            num_reqs,
            cudagraph_runtime_mode,
            batch_desc_num_reqs,
            batch_desc_uniform,
        )

    def compute_draft_slot_mapping(*, cad: Any, **kwargs: Any) -> Any:
        num_query_rows = len(cad.query_start_loc) - 1
        if len(cad.block_table_tensor) != num_query_rows:
            cad = cad.replace(block_table_tensor=cad.block_table_tensor[:num_query_rows])
        return original_compute_slot_mapping(cad=cad, **kwargs)

    def run_compact_piecewise_draft(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not getattr(self, "_compact_piecewise_enabled", False):
            return original_run_merged_draft(self, *args, **kwargs)

        original_model = self.model
        original_use_cuda_graph = self.use_cuda_graph
        self.model = CompactPiecewiseDraftModel(self, original_model)
        self.use_cuda_graph = False
        try:
            return original_run_merged_draft(self, *args, **kwargs)
        finally:
            self.use_cuda_graph = original_use_cuda_graph
            self.model = original_model

    def compact_attn_update_stack_num_spec_norm(
        self: Any,
        draft_index: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        bound = _bind_host_attn_update(
            original_attn_update,
            self,
            draft_index,
            args,
            kwargs,
        )
        batch_size = bound.arguments["batch_size"]
        aclgraph_runtime_mode = bound.arguments["aclgraph_runtime_mode"]
        if (
            getattr(self, "_compact_piecewise_enabled", False)
            and aclgraph_runtime_mode == CUDAGraphMode.PIECEWISE
        ):
            runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=batch_size,
                uniform_decode=False,
                has_lora=False,
            )
            if runtime_mode == CUDAGraphMode.PIECEWISE:
                bound.arguments["input_batch_size"] = batch_descriptor.num_tokens

        return original_attn_update(*bound.args, **bound.kwargs)

    def draft_propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        common_attn_metadata = kwargs.get("common_attn_metadata")
        batch_size = (
            common_attn_metadata.batch_size()
            if common_attn_metadata is not None
            else settings.max_num_seqs
        )
        use_merged_full = settings.use_merged_full or (
            settings.merged_full_max_batch > 0 and batch_size <= settings.merged_full_max_batch
        )
        if use_merged_full:
            previous_padding = getattr(self.runner, "_draft_merged_full_padding", False)
            self.runner._draft_merged_full_padding = True
            try:
                return original_draft_propose(self, *args, **kwargs)
            finally:
                self.runner._draft_merged_full_padding = previous_padding

        dispatcher = self.runner.cudagraph_dispatcher
        original_dispatch = dispatcher.dispatch

        def dispatch_nonuniform(*dispatch_args: Any, **dispatch_kwargs: Any) -> Any:
            dispatch_kwargs["uniform_decode"] = False
            return original_dispatch(*dispatch_args, **dispatch_kwargs)

        dispatcher.dispatch = dispatch_nonuniform
        self._compact_piecewise_enabled = True
        try:
            return original_draft_propose(self, *args, **kwargs)
        finally:
            self._compact_piecewise_enabled = False
            dispatcher.dispatch = original_dispatch

    AscendDraftModelProposer.__init__ = init_draft_proposer
    ascend_base_proposer.AscendSpecDecodeBaseProposer.dummy_run = draft_dummy_run
    NPUModelRunner._pad_query_start_loc_for_fia = pad_query_start_loc_for_fia
    ascend_base_proposer.compute_new_slot_mapping = compute_draft_slot_mapping
    ascend_base_proposer.AscendSpecDecodeBaseProposer._run_merged_draft = (
        run_compact_piecewise_draft
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm = (
        compact_attn_update_stack_num_spec_norm
    )
    ascend_base_proposer.AscendSpecDecodeBaseProposer._propose = draft_propose
    setattr(AscendDraftModelProposer, PATCH_MARKER, True)
    return True
