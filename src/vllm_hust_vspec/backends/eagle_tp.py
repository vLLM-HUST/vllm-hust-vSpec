"""Run a replicated EAGLE drafter beside a tensor-parallel target."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any

import torch

PATCH_MARKER = "_vllm_hust_vspec_replicated_draft_tp_patched"


def _uses_replicated_draft(proposer: Any) -> bool:
    return (
        proposer.vllm_config.parallel_config.tensor_parallel_size
        != proposer.speculative_config.draft_tensor_parallel_size
    )


def _capture_draft_tp_group(proposer: Any) -> None:
    if not _uses_replicated_draft(proposer):
        proposer._vspec_draft_tp_group = None
        return
    context = proposer.tp_group_context
    context_args = getattr(context, "args", ())
    if not context_args:
        raise RuntimeError("unable to recover the replicated draft TP group")
    proposer._vspec_draft_tp_group = context_args[0]


def _draft_tp_context(proposer: Any, patch_tensor_parallel_group: Any):
    draft_group = getattr(proposer, "_vspec_draft_tp_group", None)
    if draft_group is None:
        return nullcontext()
    return patch_tensor_parallel_group(draft_group)


def _install_full_lm_head(
    proposer: Any,
    target_model: Any,
    patch_tensor_parallel_group: Any,
) -> None:
    if proposer.method != "eagle" or not _uses_replicated_draft(proposer):
        return

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
    )

    target_head = getattr(target_model, "lm_head", None)
    local_weight = getattr(target_head, "weight", None)
    if not isinstance(local_weight, torch.Tensor):
        raise RuntimeError("replicated EAGLE draft requires a target LM head")

    target_tp_group = proposer._vspec_target_tp_group
    full_weight = target_tp_group.all_gather(local_weight, dim=0)
    vocab_size = int(target_head.num_embeddings)
    full_weight = full_weight[:vocab_size].contiguous()
    if full_weight.shape != (vocab_size, int(target_head.embedding_dim)):
        raise RuntimeError(
            f"gathered EAGLE LM head has an unexpected shape: {tuple(full_weight.shape)}"
        )

    with _draft_tp_context(proposer, patch_tensor_parallel_group):
        draft_head = ParallelLMHead(
            vocab_size,
            int(target_head.embedding_dim),
            bias=False,
            params_dtype=local_weight.dtype,
            org_num_embeddings=int(target_head.org_vocab_size),
            padding_size=int(target_head.padding_size),
            quant_config=None,
            prefix="lm_head",
        )
    draft_head.weight = torch.nn.Parameter(full_weight, requires_grad=False)
    proposer.model.lm_head = draft_head


def _wrap_method_in_draft_tp_group(
    proposer_cls: type,
    method_name: str,
    patch_tensor_parallel_group: Any,
) -> None:
    original_method = getattr(proposer_cls, method_name)

    @wraps(original_method)
    def wrapped(self: Any, *args: Any, **kwargs: Any):
        with _draft_tp_context(self, patch_tensor_parallel_group):
            return original_method(self, *args, **kwargs)

    setattr(proposer_cls, method_name, wrapped)


def apply_replicated_draft_tp_patch() -> bool:
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    try:
        from vllm_ascend.spec_decode.utils import patch_tensor_parallel_group
    except ImportError:
        from vllm_ascend.spec_decode.llm_base_proposer import (
            patch_tensor_parallel_group,
        )

    if getattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, False):
        return False

    original_init = AscendSpecDecodeBaseProposer.__init__
    original_share_lm_head = AscendSpecDecodeBaseProposer._maybe_share_lm_head

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        from vllm.distributed.parallel_state import get_tp_group

        self._vspec_target_tp_group = get_tp_group()
        _capture_draft_tp_group(self)

    def share_lm_head(self: Any, model: Any) -> None:
        original_share_lm_head(self, model)
        _install_full_lm_head(self, model, patch_tensor_parallel_group)

    AscendSpecDecodeBaseProposer.__init__ = init
    AscendSpecDecodeBaseProposer._maybe_share_lm_head = share_lm_head

    # Model construction also reads the process-global TP group. Both host ABIs
    # therefore need the draft group while loading the out-of-tree model.
    _wrap_method_in_draft_tp_group(
        AscendSpecDecodeBaseProposer,
        "_get_model",
        patch_tensor_parallel_group,
    )

    original_runtime_context = getattr(
        AscendSpecDecodeBaseProposer,
        "_draft_runtime_compilation_context",
        None,
    )
    if original_runtime_context is not None:

        @contextmanager
        def runtime_context(self: Any):
            with (
                _draft_tp_context(self, patch_tensor_parallel_group),
                original_runtime_context(self),
            ):
                yield

        AscendSpecDecodeBaseProposer._draft_runtime_compilation_context = runtime_context
    else:
        # Current vLLM-Ascend inlines dummy/propose execution and no longer
        # exposes the legacy runtime context hook.
        for method_name in ("dummy_run", "_propose"):
            _wrap_method_in_draft_tp_group(
                AscendSpecDecodeBaseProposer,
                method_name,
                patch_tensor_parallel_group,
            )

    setattr(AscendSpecDecodeBaseProposer, PATCH_MARKER, True)
    return True
