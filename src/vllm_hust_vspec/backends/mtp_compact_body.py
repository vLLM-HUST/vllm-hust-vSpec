"""Private proposal-only row compaction; native FIA ownership stays unchanged.

The initial MTP pass backfills every target query row as before. Later passes
only compute live request rows, padding Q/K/V at the native attention boundary
so the original captured FIA tasks, CPU metadata and update order remain intact.
Target modules, sampling and rejection are never selected by this scoped hook.
"""

import copy
import os
from contextvars import ContextVar
from functools import wraps

_GEOMETRY = ContextVar('q35_mtp_compact_body_geometry', default=None)


def cache_metadata_for_compact_rows(metadata, compact):
    """Only omit synthetic cache-write rows; retain native FIA geometry.

    Dummy capture reuses oversized first-pass metadata. Real continuation
    builders already set num_actual_tokens to the actual request count.
    The source metadata is never mutated, and all real prefix slots remain.
    """
    if metadata.num_actual_tokens <= compact:
        return metadata
    result = copy.copy(metadata)
    result.num_actual_tokens = compact
    result.slot_mapping = metadata.slot_mapping[:compact]
    return result


def check_live_continuation_counts(metadata_steps, compact):
    for per_layer in metadata_steps[1:]:
        for metadata in per_layer.values():
            if metadata.num_actual_tokens > compact:
                raise RuntimeError('Live MTP continuation exceeds the captured compact request rows')


def continuation_allowed(proposer, context, outer, compact):
    descriptor = getattr(context, 'batch_descriptor', None)
    return bool(
        proposer.method == 'mtp'
        and proposer.num_speculative_tokens in (2, 4, 6, 8)
        and 0 < compact < outer
        and getattr(descriptor, 'uniform', False)
        and getattr(descriptor, 'num_reqs', None) == compact
        and getattr(getattr(context, 'cudagraph_runtime_mode', None), 'name', '') == 'FULL'
        and not getattr(proposer, '_share_mtp_indices', False)
        and not proposer.model_returns_tuple()
        and not proposer.use_sequence_parallel_moe
    )


class CompactBodyModel:
    def __init__(self, proposer, model, compact, outer, get_context, extra):
        self.proposer, self.wrapped = proposer, model
        self.compact, self.outer = compact, outer
        self.get_context, self.extra, self.calls = get_context, extra, 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return self.wrapped(*args, **kwargs)
        context = self.get_context()
        if not continuation_allowed(self.proposer, context, self.outer, self.compact):
            return self.wrapped(*args, **kwargs)
        kwargs = dict(kwargs)
        for name in ('input_ids', 'hidden_states', 'inputs_embeds'):
            if kwargs.get(name) is not None:
                kwargs[name] = kwargs[name][:self.compact]
        kwargs['positions'] = self.proposer._get_positions(self.compact)
        old_num, old_pad = context.num_tokens, context.padded_num_tokens
        old_extra = self.extra.num_tokens
        context.num_tokens = context.padded_num_tokens = self.compact
        self.extra.num_tokens = self.compact
        token = _GEOMETRY.set((self.compact, self.outer))
        try:
            result = self.wrapped(*args, **kwargs)
            assert result.shape[0] == self.compact
            key = (self.proposer.num_speculative_tokens, self.outer, self.compact)
            seen = self.proposer._vspec_compact_body_shapes
            if key not in seen:
                seen.add(key)
                print(f'[q35-compact-body] pid={os.getpid()} gamma={key[0]} '
                      f'outer={key[1]} body_rows={key[2]} first_pass_unchanged=True '
                      'native_FIA_query_rows_unchanged=True target_unchanged=True', flush=True)
            return result
        finally:
            _GEOMETRY.reset(token)
            self.extra.num_tokens = old_extra
            context.num_tokens, context.padded_num_tokens = old_num, old_pad


def install_compact_mtp_body():
    # Lazy imports keep the geometry/admission tests CPU-only.
    import torch
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

    from .mtp import _is_qwen35_mtp_config

    original = AscendEagleProposer._run_merged_draft
    if getattr(original, '_q35_compact_mtp_body', False):
        return False
    original_attention = AscendAttentionBackendImpl.forward

    @wraps(original_attention)
    def attention(self, layer, query, key, value, kv_cache, attn_metadata,
                  output=None, output_scale=None, output_block_scale=None):
        geometry = _GEOMETRY.get()
        if geometry is None:
            return original_attention(self, layer, query, key, value, kv_cache, attn_metadata,
                                      output, output_scale, output_block_scale)
        compact, outer = geometry
        assert _EXTRA_CTX.is_draft_model, 'Proposal scope leaked into target attention'
        assert query.shape[0] == compact and output is not None
        assert key is not None and value is not None
        assert attn_metadata.actual_seq_lengths_q[-1] == outer
        attn_metadata = cache_metadata_for_compact_rows(attn_metadata, compact)
        assert attn_metadata.num_actual_tokens <= compact
        assert output_scale is None and output_block_scale is None

        def pad(tensor):
            # Device zero/copy operations become ordinary nodes in the enclosing
            # graph. Real requests retain their original prefix row positions.
            padded = torch.zeros((outer, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
            padded[:compact].copy_(tensor)
            return padded

        padded_output = torch.empty((outer, *output.shape[1:]), device=output.device, dtype=output.dtype)
        original_attention(self, layer, pad(query), pad(key), pad(value), kv_cache, attn_metadata,
                           padded_output, output_scale, output_block_scale)
        output.copy_(padded_output[:compact])
        return output

    @wraps(original)
    def run(self, num_input_tokens, batch_size, *args, **kwargs):
        context = get_forward_context()
        if not hasattr(self, '_vspec_compact_body_admissions'):
            self._vspec_compact_body_admissions = set()
        admission = (self.num_speculative_tokens, num_input_tokens, batch_size,
                     getattr(context.batch_descriptor, 'uniform', False),
                     getattr(context.batch_descriptor, 'num_reqs', None))
        if admission not in self._vspec_compact_body_admissions:
            self._vspec_compact_body_admissions.add(admission)
            print(f'[q35-compact-admission] pid={os.getpid()} shape={admission} '
                  f'allowed={continuation_allowed(self, context, num_input_tokens, batch_size)} '
                  f'descriptor={context.batch_descriptor} '
                  f'sequence_parallel={self.use_sequence_parallel_moe}', flush=True)
        if (type(self) is not AscendEagleProposer or not _is_qwen35_mtp_config(self.vllm_config)
                or not continuation_allowed(self, context, num_input_tokens, batch_size)):
            return original(self, num_input_tokens, batch_size, *args, **kwargs)
        if not hasattr(self, '_vspec_compact_body_shapes'):
            self._vspec_compact_body_shapes = set()
        previous = self.model
        self.model = CompactBodyModel(self, previous, batch_size, num_input_tokens,
                                      get_forward_context, _EXTRA_CTX)
        try:
            return original(self, num_input_tokens, batch_size, *args, **kwargs)
        finally:
            self.model = previous

    run._q35_compact_mtp_body = True
    original_update = AscendEagleProposer._update_full_graph_params

    @wraps(original_update)
    def update(self, context, num_tokens, draft_attn_metadatas=None):
        compact = getattr(context.batch_descriptor, 'num_reqs', None)
        if (compact is not None and _is_qwen35_mtp_config(self.vllm_config)
                and continuation_allowed(self, context, num_tokens, compact)):
            assert draft_attn_metadatas is not None
            check_live_continuation_counts(draft_attn_metadatas, compact)
            if not hasattr(self, '_vspec_compact_body_live_counts'):
                self._vspec_compact_body_live_counts = {}
            counts = self._vspec_compact_body_live_counts
            gamma = self.num_speculative_tokens
            counts[gamma] = counts.get(gamma, 0) + 1
            if counts[gamma] in (1, 2, 32, 128, 1024):
                print(f'[q35-compact-live] pid={os.getpid()} gamma={gamma} '
                      f'outer={num_tokens} rows={compact} checks={counts[gamma]} '
                      'all_live_KV_rows_preserved=True', flush=True)
        return original_update(self, context, num_tokens, draft_attn_metadatas)

    AscendAttentionBackendImpl.forward = attention
    AscendEagleProposer._run_merged_draft = run
    AscendEagleProposer._update_full_graph_params = update
    return True
