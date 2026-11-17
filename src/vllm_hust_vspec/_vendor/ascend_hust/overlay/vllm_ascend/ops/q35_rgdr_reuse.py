"""Opt-in native-arithmetic RGDR buffer reuse with full-state replay shadows.

Only the target's already-admitted pure-spec fourteen-request fused-input
lane is selected. Other batches, mixed prefills and the draft remain native.
The direct kernel is independently built, never installed over CANN.
"""

import os
from pathlib import Path

import torch


PRIMITIVE = Path(os.environ.get('VSPEC_RGDR_ARTIFACT_DIR', '/__vspec_artifacts_not_configured__'))


def eligible(meta, inputs, state, lengths, indices, accepted, fused):
    q, k, v, g, beta = inputs
    tokens = q.shape[0]
    return bool(fused and not meta.num_prefills and not meta.num_decodes
        and meta.num_spec_decodes == 14 and meta.spec_sequence_masks is not None
        and tokens in (70, 98) and q.shape == k.shape == (tokens, 8, 128)
        and v.shape == (tokens, 16, 128) and g.shape == beta.shape == (tokens, 16)
        and state.ndim == 4 and state.shape[0] >= tokens
        and state.shape[1:] == (16, 128, 128)
        and lengths.shape == (15,) and accepted.shape == (14,) and indices.shape == (tokens,)
        and q.dtype == k.dtype == v.dtype == beta.dtype == torch.bfloat16
        and state.dtype == g.dtype == torch.float32
        and lengths.dtype == indices.dtype == accepted.dtype == torch.int32
        and all(t.is_contiguous() and t.device == q.device
                for t in (*inputs, state, lengths, indices, accepted)))


def _fake_full128(query, key, value, beta, state, lengths, indices, g, accepted, v_step, output_buffers):
    return torch.empty_like(value)


class StateShadowWorkspace:
    def __init__(self):
        self.state = self.differences = None

    def acquire(self, state):
        if self.state is None:
            self.state = torch.empty_like(state)
            self.differences = torch.empty_like(state, dtype=torch.bool)
        if (self.state.shape != state.shape or self.state.dtype != state.dtype
                or self.state.device != state.device):
            raise RuntimeError('RGDR full-bank shadow geometry changed')
        return self.state, self.differences


class RGDRReuse:
    def __init__(self, attention, mode, workspace):
        self.attention, self.mode, self.workspace = attention, mode, workspace
        self.captured = set()
        self.counts = torch.zeros(209, dtype=torch.int64, device=attention.A_log.device)
        self.errors = torch.zeros((209, 2), dtype=torch.int32, device=attention.A_log.device)

    def apply(self, meta, inputs, state, lengths, indices, accepted, fused):
        if not eligible(meta, inputs, state, lengths, indices, accepted, fused):
            return None
        q, k, v, g, beta = inputs
        tokens = q.shape[0]
        if self.mode == 'validate':
            reference_state, differences = self.workspace.acquire(state)
            reference_state.copy_(state)
        actual = torch.ops.vspec_rgdr_pipeline.project(q, k, v, beta, state, lengths, indices, g, accepted, 64, 2)
        if self.mode == 'validate':
            reference = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=q, key=k, value=v, g=g, beta=beta, state=reference_state,
                scale=128 ** -0.5, actual_seq_lengths=lengths,
                ssm_state_indices=indices, num_accepted_tokens=accepted)
            torch.ne(state.view(torch.int32), reference_state.view(torch.int32), out=differences)
            flags = torch.stack(((actual.view(torch.int16) != reference.view(torch.int16)).any(),
                                 differences.any())).to(torch.int32)
            torch.maximum(self.errors[tokens], flags, out=self.errors[tokens])
            self.counts[tokens].add_(1)
        if tokens not in self.captured:
            self.captured.add(tokens)
            print(f'[q35-rgdr-reuse] pid={os.getpid()} layer={self.attention.prefix} '
                  f'mode={self.mode} tokens={tokens} requests=14 all_intermediate_states_written=True', flush=True)
        return actual

    def reset(self):
        self.counts.zero_()
        self.errors.zero_()

    def metrics(self):
        counts, flags = self.counts.cpu().tolist(), self.errors.cpu().tolist()
        return dict(mode=self.mode, captured_shapes=sorted(self.captured),
                    actual_validation_counts={str(t): n for t, n in enumerate(counts) if n},
                    output_and_full_FP32_state_bitwise_error_flags={str(t): flags[t]
                                                                  for t, n in enumerate(counts) if n},
                    full_bank_including_intermediate_and_unused_slots=True,
                    on_mode_replay_counts_NOT_instrumented=self.mode == 'on',
                    stats_read_after_timers_only=True)


class RGDRReuseManager:
    def __init__(self, runner, mode):
        if mode not in ('off', 'validate', 'on'):
            raise ValueError('RGDR reuse mode must be off, validate, or on')
        self.runner, self.mode, self.layers = runner, mode, {}
        self.workspace = StateShadowWorkspace()

    def bind_model(self):
        if self.mode == 'off':
            return
        if self.runner._q35_gdn_input_manager.mode != self.mode:
            raise RuntimeError('RGDR reuse requires matching admitted GDN input fusion')
        torch.ops.load_library(str(PRIMITIVE / 'build-r1/lib/libvspec_rgdr_adapter.so'))
        torch.library.register_fake('vspec_rgdr_pipeline::project')(_fake_full128)
        for prefix, input_helper in self.runner._q35_gdn_input_manager.layers.items():
            helper = RGDRReuse(input_helper.attention, self.mode, self.workspace)
            input_helper.attention._q35_rgdr_reuse = helper
            self.layers[prefix] = helper
        if len(self.layers) != 30:
            raise RuntimeError('RGDR reuse requires all thirty pinned target GDN layers')

    def reset(self):
        for helper in self.layers.values():
            helper.reset()

    def attach_metrics(self, records):
        if self.mode != 'off':
            for record in records:
                record['rgdr_reuse'] = self.layers[record['layer']].metrics()
        return records
