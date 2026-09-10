"""Device-event ordering for EAGLE FULL graph parameter updates."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context

from ..host_compat import call_with_supported_kwargs

PATCH_MARKER = "_vllm_hust_vspec_event_ordering_patched"


def _host_has_native_event_ordering(
    graph_wrapper: type[Any],
    model_runner: type[Any],
    proposer: type[Any],
) -> bool:
    if not hasattr(graph_wrapper, "wait_for_prior_replay_before_update"):
        return False
    try:
        graph_source = inspect.getsource(graph_wrapper.__call__)
        target_source = inspect.getsource(model_runner._update_full_graph_params_if_needed)
        draft_source = inspect.getsource(proposer._update_full_graph_params)
    except (OSError, TypeError):
        return False
    return (
        "event_ordered_replay" in graph_source
        and "wait_for_prior_replay_before_update" in target_source
        and "wait_for_prior_replay_before_update" in draft_source
    )


def _call_host_graph_init(
    original_init: Any,
    instance: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    call_with_supported_kwargs(original_init, instance, *args, **kwargs)


def apply_graph_event_ordering_patch() -> bool:
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(ACLGraphWrapper, PATCH_MARKER, False):
        return False
    if _host_has_native_event_ordering(
        ACLGraphWrapper,
        NPUModelRunner,
        AscendSpecDecodeBaseProposer,
    ):
        setattr(ACLGraphWrapper, PATCH_MARKER, True)
        return False

    original_init = ACLGraphWrapper.__init__
    original_call = ACLGraphWrapper.__call__
    original_target_update = NPUModelRunner._update_full_graph_params_if_needed
    original_draft_update = AscendSpecDecodeBaseProposer._update_full_graph_params

    def graph_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _call_host_graph_init(original_init, self, *args, **kwargs)
        self.event_ordered_replay = self.runtime_mode == CUDAGraphMode.FULL
        self._pending_update_dependency_event = None

    def wait_for_prior_replay_before_update(
        self: Any,
        update_stream: torch.npu.Stream,
    ) -> None:
        event = self._pending_update_dependency_event
        self._pending_update_dependency_event = None
        if event is not None:
            update_stream.wait_event(event)

    def graph_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.event_ordered_replay:
            return original_call(self, *args, **kwargs)

        forward_context = get_forward_context()
        if (
            forward_context.cudagraph_runtime_mode != self.runtime_mode
            or self.runtime_mode != CUDAGraphMode.FULL
        ):
            return original_call(self, *args, **kwargs)

        entry = self.concrete_aclgraph_entries.get(forward_context.batch_descriptor)
        if entry is None or entry.aclgraph is None:
            return original_call(self, *args, **kwargs)

        self._pending_update_dependency_event = getattr(
            entry,
            "last_replay_done_event",
            None,
        )
        original_enable_enpu = self.enable_enpu
        self.enable_enpu = True
        try:
            output = original_call(self, *args, **kwargs)
        finally:
            self.enable_enpu = original_enable_enpu

        events = getattr(entry, "replay_done_events", None)
        if events is None:
            events = (torch.npu.Event(), torch.npu.Event())
            entry.replay_done_events = events
            entry.next_replay_event_index = 0
        event_index = entry.next_replay_event_index
        entry.next_replay_event_index ^= 1
        replay_done_event = events[event_index]
        replay_done_event.record()
        entry.last_replay_done_event = replay_done_event
        return output

    def target_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self.model, ACLGraphWrapper):
            self.model.wait_for_prior_replay_before_update(self.update_stream)
        return original_target_update(self, *args, **kwargs)

    def draft_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self._runnable, ACLGraphWrapper):
            self._runnable.wait_for_prior_replay_before_update(self.update_stream)
        return original_draft_update(self, *args, **kwargs)

    ACLGraphWrapper.__init__ = graph_init
    ACLGraphWrapper.__call__ = graph_call
    ACLGraphWrapper.wait_for_prior_replay_before_update = wait_for_prior_replay_before_update
    NPUModelRunner._update_full_graph_params_if_needed = target_update
    AscendSpecDecodeBaseProposer._update_full_graph_params = draft_update
    setattr(ACLGraphWrapper, PATCH_MARKER, True)
    return True
