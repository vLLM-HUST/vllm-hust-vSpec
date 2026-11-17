"""Exact immutable weight-layout reuse with in-graph native conv/state shadows."""

import os

import torch


class ConvShadowWorkspace:
    """One serial-worker scratch, never a shortened persistent-state sample.

    Each captured sequence is copy -> candidate/native -> full-state compare
    -> flag reduction before the next layer reuses this storage. The pinned
    single-stream runner and twin graph owner keep replays FIFO as well.
    """

    def __init__(self):
        self.state = self.differences = None

    def acquire(self, state):
        if self.state is None:
            self.state = torch.empty_like(state)
            self.differences = torch.empty_like(state, dtype=torch.bool)
        if (self.state.shape != state.shape or self.state.dtype != state.dtype
                or self.state.device != state.device):
            raise RuntimeError('Serial full-state shadow geometry changed')
        return self.state, self.differences


class CachedConvWeights:
    def __init__(self, attention, mode, shadow_workspace=None):
        self.attention, self.mode = attention, mode
        self.source = attention.conv1d.weight
        if (self.source.dtype != torch.bfloat16 or tuple(self.source.shape) != (4096, 1, 4)
                or not self.source.is_contiguous()):
            raise RuntimeError('Cached convolution requires the pinned BF16 Q35 TP2 weight geometry')
        self.address = self.source.data_ptr()
        try:
            self.version = self.source._version
        except RuntimeError:
            # Serving inference tensors have no version counter. Their
            # immutable load-time owner is still checked on every capture.
            self.version = None
        self.weights = self.source.view(4096, 4).transpose(0, 1).contiguous()
        # The existing input-fusion counter belongs to the SAME pure-spec
        # graph and avoids introducing another hot per-layer counter kernel.
        self.input_fusion = attention._q35_input_fusion
        self.errors = torch.zeros((209, 2), dtype=torch.int32, device=self.source.device)
        self.shadow_workspace = shadow_workspace if shadow_workspace is not None else ConvShadowWorkspace()
        self.captured = set()

    def apply(self, output, value, native_weights, state, bias, query, indices, accepted, meta, activation):
        if (meta.num_prefills or meta.num_decodes or meta.num_spec_decodes <= 0
                or value.ndim != 2 or value.shape[1] != 4096 or value.shape[0] > 208
                or value.shape[0] not in (meta.num_spec_decodes * 5, meta.num_spec_decodes * 7)
                or value.dtype != torch.bfloat16 or state.dtype != torch.bfloat16
                or tuple(state.shape[-2:]) != (9, 4096)):
            return False
        if (self.source is not self.attention.conv1d.weight or self.source.data_ptr() != self.address
                or (self.version is not None and self.source._version != self.version)):
            raise RuntimeError('Immutable convolution weights changed after caching')
        tokens = value.shape[0]
        if self.mode == 'validate':
            reference_state, state_differences = self.shadow_workspace.acquire(state)
            reference_state.copy_(state)
        def invoke(destination, weights, cache):
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                destination, value, weights, conv_state=cache, bias_opt=bias,
                query_start_loc_opt=query, cache_indices_opt=indices, initial_state_mode_opt=None,
                num_accepted_tokens_opt=accepted, activation_mode=activation, pad_slot_id=-1, run_mode=1)
        invoke(output, self.weights, state)
        if self.mode == 'validate':
            reference = torch.empty_like(output)
            invoke(reference, native_weights, reference_state)
            # Integer views also compare untouched NaN payloads and signed
            # zeros in unused physical cache slots, not just real outputs.
            torch.ne(state.view(torch.int16), reference_state.view(torch.int16), out=state_differences)
            differences = torch.stack(((output.view(torch.int16) != reference.view(torch.int16)).any(),
                state_differences.any())).to(torch.int32)
            torch.maximum(self.errors[tokens], differences, out=self.errors[tokens])
        if tokens not in self.captured:
            self.captured.add(tokens)
            print(f'[q35-cached-conv] pid={os.getpid()} layer={self.attention.prefix} mode={self.mode} '
                  f'tokens={tokens} cached_weight_contiguous=True native_math=True', flush=True)
        return True

    def reset(self):
        self.errors.zero_()

    def metrics(self):
        counts = self.input_fusion.counts.cpu().tolist()
        flags = self.errors.cpu().tolist()
        return dict(mode=self.mode, captured_shapes=sorted(self.captured),
            same_graph_input_fusion_replay_counts={str(t): counts[t] for t in self.captured if counts[t]},
            output_and_full_state_error_flags={str(t): flags[t] for t in self.captured if counts[t]},
            error_comparison_is_bitwise_int16=True, counter_piggybacks_same_graph_input_fusion=True,
            full_state_shadow_workspace_shared_serially=True)


class CachedConvManager:
    def __init__(self, runner, mode):
        if mode not in ('off', 'validate', 'on'):
            raise ValueError('Cached conv mode must be off, validate, or on')
        self.runner, self.mode, self.layers = runner, mode, {}
        self.shadow_workspace = ConvShadowWorkspace()

    def bind_model(self, model):
        if self.mode == 'off':
            return
        if self.runner._q35_gdn_input_manager.mode != self.mode:
            raise RuntimeError('Cached conv requires the matching admitted GDN input-fusion scope')
        for prefix, input_helper in self.runner._q35_gdn_input_manager.layers.items():
            helper = CachedConvWeights(input_helper.attention, self.mode, self.shadow_workspace)
            input_helper.attention._q35_cached_conv = helper
            self.layers[prefix] = helper
        if len(self.layers) != 30:
            raise RuntimeError('Cached conv requires all thirty pinned native GDN layers')

    def reset(self):
        for helper in self.layers.values():
            helper.reset()

    def attach_metrics(self, records):
        if self.mode != 'off':
            for record in records:
                record['cached_conv'] = self.layers[record['layer']].metrics()
        return records
