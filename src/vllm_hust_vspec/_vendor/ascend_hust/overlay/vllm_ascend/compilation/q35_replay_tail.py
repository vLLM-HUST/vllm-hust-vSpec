"""Opt-in previous-replay fence, not removal of graph-update ordering.

All host-side task updates still follow their completed previous graph.
Target and merged-draft wrappers own independent end-of-replay fences. The
native target full-stream barrier used to protect both kinds of task handles;
shortening it therefore also requires an explicit previous-draft fence.
The native per-attention ExternalEvent wait/reset/record protocol is intact.
Only unrelated, later same-stream preparation work may remain in flight.
Unknown paths invalidate the fence and retain the original full-stream wait.
"""

import os
import threading
from collections import Counter

import torch

from vllm_ascend import envs


def tail_eligible(config, full, use_eagle, enable_enpu):
    parallel = config.parallel_config
    return bool(full and not enable_enpu
                and getattr(config.speculative_config, 'method', None) == 'mtp'
                and config.scheduler_config.async_scheduling
                and config.cache_config.mamba_cache_mode == 'align'
                and config.lora_config is None
                and getattr(config, 'kv_transfer_config', None) is None
                and getattr(config.model_config.hf_text_config, 'model_type', '')
                in ('qwen3_5', 'qwen3_5_moe', 'qwen3_5_text', 'qwen3_5_moe_text')
                and parallel.tensor_parallel_size == 2
                and parallel.pipeline_parallel_size == 1
                and parallel.data_parallel_size == 1
                and getattr(parallel, 'decode_context_parallel_size', 1) == 1
                and getattr(parallel, 'prefill_context_parallel_size', 1) == 1)


class Q35ReplayTail:
    def __init__(self, config, full, use_eagle, enable_enpu):
        self.mode = envs.VLLM_ASCEND_Q35_REPLAY_TAIL
        # use_eagle also applies to the TARGET wrapper in Ascend MTP. Bind
        # observational labels from the actual per-call forward context.
        self.role = 'unbound'
        if self.mode not in ('off', 'validate', 'on'):
            raise ValueError('Q35 replay-tail mode must be off, validate, or on')
        self.enabled = self.mode != 'off' and tail_eligible(config, full, use_eagle, enable_enpu)
        self.event = self.stream_key = self.owner = None
        self.pending = False
        self.counts = Counter()
        if self.mode != 'off' and full:
            print(f'[q35-replay-tail-init] pid={os.getpid()} mode={self.mode} role={self.role} enabled={self.enabled}', flush=True)

    def bind_role(self, is_draft):
        # Logging only: never changes eligibility, waits, events or tensors.
        self.role = 'draft' if is_draft else 'target'

    def invalidate(self):
        # Captures, eager frames and mismatching wrappers are not represented
        # by the last replay event. The next replay takes the native barrier.
        self.pending = False

    def before(self, stream, descriptor):
        if not self.enabled:
            return False
        identity = (stream.npu_stream, str(stream.device))
        owner = threading.get_ident()
        if self.owner is not None and (self.owner != owner or self.stream_key != identity):
            if self.pending:
                self.event.synchronize()
            self.invalidate()
            self.enabled = False
            self.counts['unsupported_owner_or_stream'] += 1
            return False
        self.owner, self.stream_key = owner, identity
        if not self.pending:
            self.counts['native_full_stream_fallback'] += 1
            return False
        # This is a host wait for the entire previous graph, across all graph
        # keys owned by this wrapper. No cached tensor values are consulted.
        self.event.synchronize()
        self.pending = False
        self.counts['previous_graph_waits'] += 1
        if self.mode == 'validate':
            assert self.event.query(), 'Previous target graph has not completed'
            self.counts['previous_graph_completion_checks'] += 1
            self.counts['later_stream_work_pending'] += int(not stream.query())
        tokens, requests = descriptor.num_tokens, descriptor.num_reqs
        if descriptor.uniform and requests and tokens % requests == 0:
            self.counts[f'width{tokens // requests}'] += 1
        total = self.counts['previous_graph_waits']
        if total in (1, 2, 128, 1024):
            print(f'[q35-replay-tail] pid={os.getpid()} mode={self.mode} role={self.role} counts={dict(self.counts)}', flush=True)
        return True

    def after(self, stream):
        if not self.enabled:
            return
        if self.event is None:
            self.event = torch.npu.Event()
        # Enqueued immediately after replay, before future preparation work.
        # Reuse happens only after before() consumed the previous generation
        # or a native full-stream barrier completed an invalidated generation.
        self.event.record(stream)
        self.pending = True
        self.counts['tail_records'] += 1
