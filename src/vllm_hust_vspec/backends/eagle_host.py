"""EAGLE proposer controls that can be injected across host revisions."""

from __future__ import annotations

import copy
import inspect
import os
from typing import Any

COMPILE_PATCH_MARKER = "_vllm_hust_vspec_compile_control_patched"
ISOLATION_PATCH_MARKER = "_vllm_hust_vspec_shared_isolation_patched"
PRESERVE_HIDDEN_PATCH_MARKER = "_vllm_hust_vspec_preserve_hidden_patched"


def _source_mentions(owner: Any, method_name: str, token: str) -> bool:
    method = getattr(owner, method_name, None)
    if method is None:
        return False
    try:
        return token in inspect.getsource(method)
    except (OSError, TypeError):
        return False


def apply_draft_compile_control_patch() -> bool:
    from vllm_ascend.spec_decode import llm_base_proposer as proposer_module

    proposer = proposer_module.AscendSpecDecodeBaseProposer
    if getattr(proposer, COMPILE_PATCH_MARKER, False):
        return False
    environment_name = "VLLM_ASCEND_EAGLE_DISABLE_DRAFT_TORCH_COMPILE"
    if (
        hasattr(proposer, "_draft_runtime_compilation_context")
        and _source_mentions(proposer, "__init__", environment_name)
        and _source_mentions(
            proposer,
            "_propose",
            "_draft_runtime_compilation_context",
        )
    ):
        setattr(proposer, COMPILE_PATCH_MARKER, True)
        return False

    original_init = proposer.__init__
    original_dummy_run = proposer.dummy_run
    original_propose = proposer._propose

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.disable_draft_torch_compile = (
            self.method == "eagle" and os.environ.get(environment_name) == "1"
        )
        if self.disable_draft_torch_compile:
            self.maybe_eager_context = proposer_module._maybe_eager_context(self.vllm_config)

    def runtime_compilation_context(self: Any) -> Any:
        if self.disable_draft_torch_compile:
            return proposer_module._maybe_eager_context(self.vllm_config)
        from contextlib import nullcontext

        return nullcontext()

    def dummy_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._draft_runtime_compilation_context():
            return original_dummy_run(self, *args, **kwargs)

    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._draft_runtime_compilation_context():
            return original_propose(self, *args, **kwargs)

    proposer.__init__ = init
    proposer._draft_runtime_compilation_context = runtime_compilation_context
    proposer.dummy_run = dummy_run
    proposer._propose = propose
    setattr(proposer, COMPILE_PATCH_MARKER, True)
    return True


def apply_shared_module_isolation_patch() -> bool:
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )

    proposer = AscendSpecDecodeBaseProposer
    if getattr(proposer, ISOLATION_PATCH_MARKER, False):
        return False
    environment_name = "VLLM_ASCEND_EAGLE_ISOLATE_SHARED_MODULES"
    if _source_mentions(proposer, "load_model", environment_name):
        setattr(proposer, ISOLATION_PATCH_MARKER, True)
        return False

    original_share_embeddings = proposer._maybe_share_embeddings

    def maybe_share_embeddings(self: Any, target_model: Any) -> Any:
        isolate = self.method == "eagle" and os.environ.get(environment_name) == "1"
        if not isolate:
            return original_share_embeddings(self, target_model)
        for module in self.model.modules():
            rotary_embedding = getattr(module, "rotary_emb", None)
            if rotary_embedding is not None:
                module.rotary_emb = copy.deepcopy(rotary_embedding)
        return None

    proposer._maybe_share_embeddings = maybe_share_embeddings
    setattr(proposer, ISOLATION_PATCH_MARKER, True)
    return True


def apply_preserve_target_hidden_patch() -> bool:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = NPUModelRunner
    if getattr(runner, PRESERVE_HIDDEN_PATCH_MARKER, False):
        return False
    environment_name = "VLLM_ASCEND_EAGLE_PRESERVE_TARGET_HIDDEN"
    if _source_mentions(runner, "execute_model", environment_name):
        setattr(runner, PRESERVE_HIDDEN_PATCH_MARKER, True)
        return False

    original_execute_model = runner.execute_model

    def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute_model(self, *args, **kwargs)
        state = self.execute_model_state
        preserve = (
            self.speculative_config is not None
            and self.speculative_config.method == "eagle"
            and os.environ.get(environment_name) == "1"
        )
        if preserve and state is not None:
            self.execute_model_state = state._replace(hidden_states=state.hidden_states.clone())
        return result

    runner.execute_model = execute_model
    setattr(runner, PRESERVE_HIDDEN_PATCH_MARKER, True)
    return True
