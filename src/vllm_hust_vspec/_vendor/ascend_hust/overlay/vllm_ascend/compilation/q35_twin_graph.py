"""Two owned native FIA graphs, each fenced before mutable task updates.

No attention arguments, GPU ordering, ExternalEvents, or model math change.
Only uniform fourteen-request MTP4/6 and one-through-three-request MTP6 FULL graphs are admitted. Each slot
owns its task handles, update events, workspace and previous-generation tail.
Unrelated shapes keep the native graph and full-stream synchronization.
"""

import os
import threading
from collections import Counter
from dataclasses import dataclass, field

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.q35_replay_tail import tail_eligible

REQUESTS = 14
REQUEST_COUNTS = (1, 2, 3, REQUESTS)
WIDTHS = (5, 7)
SLOTS = 2
SELECTION_ATTR = '_q35_twin_graph_selection'


def descriptor_admitted(descriptor):
    return bool(descriptor is not None and descriptor.uniform
                and descriptor.num_reqs in REQUEST_COUNTS
                and descriptor.num_tokens in tuple(descriptor.num_reqs * width for width in WIDTHS)
                and (descriptor.num_reqs == REQUESTS or descriptor.num_tokens == descriptor.num_reqs * 7))


def empty_params_like(source):
    return type(source)(events={key: [] for key in source.events},
                        workspaces={key: None for key in source.workspaces},
                        handles={key: [] for key in source.handles},
                        attn_params={key: [] for key in source.attn_params})


@dataclass
class Slot:
    source: object
    descriptor: object
    role: str
    number: int
    params: object
    entries: dict = field(default_factory=dict)
    tail: object = None
    pending: bool = False
    verified: bool = False
    replays: int = 0


def route_params(source, role):
    """Resolve against the original width/gamma-specific source, never global IDs alone."""
    if source is None:
        return source
    try:
        context = get_forward_context()
    except AssertionError:
        return source
    selection = getattr(context, SELECTION_ATTR, None)
    if (selection is not None and selection.source is source and selection.role == role
            and selection.descriptor == context.batch_descriptor):
        return selection.params
    return source


class Q35TwinGraph:
    def __init__(self, config, full, use_eagle, enable_enpu):
        self.mode = envs.VLLM_ASCEND_Q35_TWIN_GRAPH
        if self.mode not in ('off', 'validate', 'on'):
            raise ValueError('Twin graph mode must be off, validate, or on')
        self.enabled = bool(self.mode != 'off'
                            and tail_eligible(config, full, use_eagle, enable_enpu)
                            and config.model_config.dtype == torch.bfloat16
                            and config.model_config.quantization is None)
        if self.enabled and envs.VLLM_ASCEND_Q35_REPLAY_TAIL != 'off':
            raise ValueError('Twin graphs and shared previous-replay-tail mode cannot compose')
        self.groups = {}
        self.next_slots = Counter()
        self.counts = Counter()
        self.owner = self.stream_key = None
        self.current = None
        if self.mode != 'off' and full:
            print(f'[q35-twin-init] pid={os.getpid()} mode={self.mode} enabled={self.enabled}', flush=True)

    def _select(self, context, slot):
        setattr(context, SELECTION_ATTR, slot)
        self.current = slot

    def _owner(self, stream):
        owner = threading.get_ident()
        identity = (stream.npu_stream, str(stream.device))
        if self.owner is not None and (self.owner != owner or self.stream_key != identity):
            # Do not silently move a mutable graph to another host/compute stream.
            raise RuntimeError(f'Twin graph owner or compute stream changed: '
                               f'owner={self.owner}->{owner} stream={self.stream_key}->{identity}')
        self.owner, self.stream_key = owner, identity

    def consume_verified_fence(self, context, entry):
        slot = self.current
        if (not self.enabled or slot is None or getattr(context, SELECTION_ATTR, None) is not slot
                or slot.descriptor != context.batch_descriptor
                or slot.entries.get(slot.descriptor) is not entry or not slot.verified):
            return False
        self._owner(torch.npu.current_stream())
        slot.verified = False
        self.counts['verified_native_barrier_replacements'] += 1
        return True

    def active_entry(self, context, entry):
        slot = self.current
        return bool(self.enabled and slot is not None
                    and getattr(context, SELECTION_ATTR, None) is slot
                    and slot.descriptor == context.batch_descriptor
                    and slot.entries.get(slot.descriptor) is entry)

    def _log(self, slot):
        if slot.replays in (1, 2, 32, 128, 256, 1024):
            tokens = slot.descriptor.num_tokens
            print(f'[q35-twin] pid={os.getpid()} mode={self.mode} role={slot.role} '
                  f'tokens={tokens} slot={slot.number} replays={slot.replays} '
                  f'handles={len(slot.params.handles[tokens])} '
                  f'events={len(slot.params.events[tokens])} '
                  f'workspace={slot.params.workspaces[tokens].data_ptr()} '
                  f'counts={dict(self.counts)}', flush=True)

    def call(self, wrapper, native, *args, **kwargs):
        if not self.enabled:
            return native(*args, **kwargs)
        context = get_forward_context()
        descriptor = context.batch_descriptor
        # Preserve eager/profile/prefill/nonuniform/tail execution unchanged.
        if (context.cudagraph_runtime_mode != wrapper.runtime_mode
                or not descriptor_admitted(descriptor)
                or _EXTRA_CTX.is_draft_model_prefill):
            setattr(context, SELECTION_ATTR, None)
            self.current = None
            return native(*args, **kwargs)
        from vllm_ascend.compilation import acl_graph

        role = 'draft' if _EXTRA_CTX.is_draft_model else 'target'
        source = acl_graph._draft_graph_params if role == 'draft' else acl_graph._graph_params
        if source is None:
            raise RuntimeError('Twin capture requires initialized native graph parameters')
        stream = torch.npu.current_stream()
        key = (id(source), descriptor, role)
        slots = self.groups.get(key)
        original_entries = wrapper.concrete_aclgraph_entries
        if slots is None:
            # The native capture validator still runs for BOTH slots. Missing
            # runtime graphs never bypass the framework's cold-capture guard.
            slots = tuple(Slot(source, descriptor, role, index, empty_params_like(source))
                          for index in range(SLOTS))
            try:
                results = []
                for slot in slots:
                    self._select(context, slot)
                    wrapper.concrete_aclgraph_entries = slot.entries
                    results.append(native(*args, **kwargs))
                    params, tokens = slot.params, descriptor.num_tokens
                    if not (params.handles[tokens] and params.events[tokens]
                            and params.workspaces[tokens] is not None):
                        raise RuntimeError('Twin slot did not capture native FIA resources')
                left, right = (slot.params for slot in slots)
                tokens = descriptor.num_tokens
                if (set(left.handles[tokens]) & set(right.handles[tokens])
                        or {id(event) for event in left.events[tokens]}
                        & {id(event) for event in right.events[tokens]}
                        or left.workspaces[tokens].data_ptr() == right.workspaces[tokens].data_ptr()):
                    raise RuntimeError('Twin mutable resources alias')
                self.groups[key] = slots
                self.counts['independent_slot_pairs_captured'] += 1
                # Capture is a framework-validated startup phase. Its RPC
                # thread/stream need not be the inference owner. Complete
                # both captures before publishing, WITHOUT binding runtime
                # ownership. No hot-path device synchronization is added.
                torch.npu.synchronize()
                self._select(context, slots[0])
                print(f'[q35-twin-capture] pid={os.getpid()} mode={self.mode} role={role} '
                      f'tokens={tokens} slots=2 independent=True', flush=True)
                return results[0]
            finally:
                wrapper.concrete_aclgraph_entries = original_entries
        # The first actual replay retains the original full-stream barrier;
        # subsequent replays must remain on this same host/compute stream.
        self._owner(stream)
        slot = slots[self.next_slots[key] % SLOTS]
        self.next_slots[key] += 1
        self._select(context, slot)
        slot.verified = False
        if slot.pending:
            slot.tail.synchronize()
            if not slot.tail.query():
                raise RuntimeError('Own previous twin graph generation is incomplete')
            slot.pending = False
            slot.verified = True
            self.counts['own_previous_graph_waits'] += 1
            if self.mode == 'validate':
                other = slots[1 - slot.number]
                self.counts['other_slot_inflight_handoffs'] += int(
                    other.pending and not other.tail.query())
        try:
            wrapper.concrete_aclgraph_entries = slot.entries
            result = native(*args, **kwargs)
            if slot.verified:
                raise RuntimeError('Native replay did not consume its own verified fence')
            if slot.tail is None:
                slot.tail = torch.npu.Event()
            slot.tail.record(stream)
            slot.pending = True
            slot.replays += 1
            self.counts['replay_tail_records'] += 1
            self._log(slot)
            return result
        finally:
            wrapper.concrete_aclgraph_entries = original_entries
