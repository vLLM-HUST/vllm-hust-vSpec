"""Native FP32 router weight cache and bit-exact BF16 shared gating."""

import os

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.fused_moe.q35_single_stream_events import skip_events
from vllm_ascend.ops.q35_lookup_cache import immutable_key


def native_weight(gate):
    return gate.weight_fp32 if hasattr(gate, 'weight_fp32') else gate.weight.to(torch.float32)


def base_dense(value):
    return value.is_contiguous() and torch_npu.get_npu_format(value) in (0, 2)


class ImmutableMoECache:
    def __init__(self, owner, role, mode):
        if not owner._q35_single_stream_events or owner.ascend_shared_experts is None:
            raise RuntimeError('Immutable MoE cache requires admitted native serial shared experts')
        self.owner, self.role, self.mode = owner, role, mode
        self.gate, self.prefix = owner.gate, owner.layer_name
        self.source = self.gate.weight
        if self.source.shape != (256, 2048) or self.source.dtype != torch.bfloat16:
            raise RuntimeError('Pinned native BF16 router geometry changed')
        self.source_key = immutable_key(self.source)
        self.original_fp32 = getattr(self.gate, 'weight_fp32', None)
        self.router_weight = native_weight(self.gate) if base_dense(self.source) else None
        if self.router_weight is not None and not base_dense(self.router_weight):
            self.router_weight = None
        self.router_counts = torch.zeros(209, dtype=torch.int64, device=self.source.device)
        self.gate_counts = torch.zeros_like(self.router_counts)
        self.errors = torch.zeros((209, 2), dtype=torch.int32, device=self.source.device)
        self.captured_router, self.captured_gate = set(), set()

    def check_source(self):
        if (self.gate.weight is not self.source or immutable_key(self.source) != self.source_key
                or getattr(self.gate, 'weight_fp32', None) is not self.original_fp32):
            raise RuntimeError('Immutable native router source changed after load')

    def router(self, value):
        self.check_source()
        tokens = value.shape[0]
        if (self.router_weight is None or not 0 < tokens <= 208 or value.shape[1] != 2048
                or value.dtype != torch.float32 or not skip_events(self.owner)):
            return F.linear(value, native_weight(self.gate))
        result = F.linear(value, self.router_weight)
        if self.mode == 'validate':
            expected = F.linear(value, native_weight(self.gate))
            flag = ((result.view(torch.int32) != expected.view(torch.int32))
                    | ~torch.isfinite(result) | ~torch.isfinite(expected)).any().to(torch.int32)
            self.router_counts[tokens].add_(1)
            torch.maximum(self.errors[tokens, 0], flag, out=self.errors[tokens, 0])
        self.captured_router.add(tokens)
        return result

    def shared_gate(self, gate, value):
        if (value.ndim != 2 or not 14 <= value.shape[0] <= 208 or value.shape[1] != 2048
                or gate.shape != (value.shape[0], 1) or value.dtype != torch.bfloat16
                or gate.dtype != torch.bfloat16 or not skip_events(self.owner)
                or not base_dense(gate) or not base_dense(value)):
            return None
        from vllm_ascend.ops.triton.q35_shared_gate import gate_multiply

        tokens = value.shape[0]
        result = gate_multiply(gate, value, block_rows=2)
        if self.mode == 'validate':
            expected = torch.sigmoid(gate) * value
            flag = ((result.view(torch.int16) != expected.view(torch.int16))
                    | ~torch.isfinite(result) | ~torch.isfinite(expected)).any().to(torch.int32)
            self.gate_counts[tokens].add_(1)
            torch.maximum(self.errors[tokens, 1], flag, out=self.errors[tokens, 1])
        self.captured_gate.add(tokens)
        return result

    @torch.inference_mode()
    def reset(self):
        self.router_counts.zero_()
        self.gate_counts.zero_()
        self.errors.zero_()

    def metrics(self):
        router, gate, errors = (value.cpu().tolist() for value in
                                (self.router_counts, self.gate_counts, self.errors))
        return dict(layer=self.prefix, role=self.role, mode=self.mode,
                    router_cache_admitted=self.router_weight is not None,
                    original_router_already_precast=self.original_fp32 is not None,
                    captured_router_tokens=sorted(self.captured_router),
                    captured_gate_tokens=sorted(self.captured_gate),
                    validation_router_counts={str(t): n for t, n in enumerate(router) if n},
                    validation_gate_counts={str(t): n for t, n in enumerate(gate) if n},
                    bitwise_error_flags={str(t): errors[t] for t in range(209) if router[t] or gate[t]},
                    on_mode_per_layer_counts_not_instrumented=self.mode == 'on',
                    native_fp32_router_linear_retained=True,
                    sigmoid_bf16_intermediate_rounding_retained=True,
                    shared_gate_block_rows=2, unsupported_layouts_native_fallback=True)


def router_linear(owner, value, gate):
    helper = getattr(owner, '_q35_immutable_cache', None)
    return F.linear(value, native_weight(gate)) if helper is None else helper.router(value)


class ImmutableMoEManager:
    def __init__(self, runner, mode):
        if mode not in ('off', 'validate', 'on'):
            raise ValueError('Immutable MoE cache requires off,validate,on')
        self.runner, self.mode, self.helpers = runner, mode, {}

    def bind_model(self):
        if self.mode == 'off':
            return
        if self.runner._q35_gdn_input_manager.mode != self.mode:
            raise RuntimeError('MoE cache requires its matching admitted BF16 Q35 input scope')
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

        for role, model in (('target', self.runner.model), ('draft', self.runner.drafter.model)):
            for _, owner in model.named_modules():
                if not isinstance(owner, AscendMoERunner):
                    continue
                helper = ImmutableMoECache(owner, role, self.mode)
                if helper.prefix in self.helpers:
                    raise RuntimeError('Duplicate native MoE cache prefix')
                owner._q35_immutable_cache = owner.ascend_shared_experts._q35_immutable_cache = helper
                self.helpers[helper.prefix] = helper
        roles = [helper.role for helper in self.helpers.values()]
        if roles.count('target') != 40 or roles.count('draft') != 1:
            raise RuntimeError('Require all forty target MoE modules and one native draft module')

    def reset(self):
        for helper in self.helpers.values():
            helper.reset()

    def attach_metrics(self, records):
        if self.mode != 'off' and records:
            records[0]['immutable_moe_cache'] = [helper.metrics() for helper in self.helpers.values()]
        return records
