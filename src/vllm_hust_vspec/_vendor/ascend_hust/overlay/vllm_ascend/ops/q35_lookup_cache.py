"""One native lookup per complete target forward; no cached token values."""

import os
import re

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


def tensor_key(value):
    return (value.data_ptr(), value.storage_offset(), tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device)


def immutable_key(value):
    try:
        version = value._version
    except RuntimeError:
        version = None
    return tensor_key(value), version


class LookupHelper:
    def __init__(self, manager, attention, role):
        self.manager, self.attention, self.role = manager, attention, role
        self.prefix = attention.attn.layer_name
        self.cache = attention.rotary_emb.cos_sin_cache
        self.q_source, self.k_source = attention.q_norm.weight, attention.k_norm.weight
        self.q_weight, self.k_weight = 1.0 + self.q_source, 1.0 + self.k_source
        self.source_keys = tuple(immutable_key(value) for value in (self.cache, self.q_source, self.k_source))
        self.counts = torch.zeros(209, dtype=torch.int64, device=self.q_source.device)
        self.errors = torch.zeros((209, 3), dtype=torch.int32, device=self.q_source.device)
        self.captured = set()

    def native(self, positions):
        return self.cache[positions].to(dtype=torch.bfloat16)

    def lookup(self, positions):
        current = (self.attention.rotary_emb.cos_sin_cache,
                   self.attention.q_norm.weight, self.attention.k_norm.weight)
        if tuple(immutable_key(value) for value in current) != self.source_keys:
            self.manager.frame = None
            raise RuntimeError('Immutable rotary/norm source changed after load')
        context = get_forward_context()
        actual, reused = self.manager.lookup(self, positions, context)
        tokens = positions.shape[-1]
        if self.manager.mode == 'validate' and 0 < tokens <= 208:
            expected = self.native(positions)
            flags = torch.stack(((actual.view(torch.int16) != expected.view(torch.int16)).any(),
                (self.q_weight.view(torch.int16) != (1.0 + self.q_source).view(torch.int16)).any(),
                (self.k_weight.view(torch.int16) != (1.0 + self.k_source).view(torch.int16)).any())).to(torch.int32)
            self.counts[tokens].add_(1)
            torch.maximum(self.errors[tokens], flags, out=self.errors[tokens])
        marker = (tokens, reused)
        if marker not in self.captured and 0 < tokens <= 208:
            self.captured.add(marker)
            print(f'[q35-native-lookup] pid={os.getpid()} mode={self.manager.mode} '
                  f'role={self.role} layer={self.prefix} tokens={tokens} '
                  f'reused={reused} native_bounds_check_retained=True', flush=True)
        return actual

    @torch.inference_mode()
    def reset(self):
        self.counts.zero_()
        self.errors.zero_()

    def metrics(self):
        counts, errors = self.counts.cpu().tolist(), self.errors.cpu().tolist()
        return dict(layer=self.prefix, role=self.role, mode=self.manager.mode,
                    captured=[dict(tokens=t, reused=r) for t, r in sorted(self.captured)],
                    validation_counts={str(t): count for t, count in enumerate(counts) if count},
                    bitwise_error_flags={str(t): errors[t] for t, count in enumerate(counts) if count},
                    on_mode_per_layer_counts_not_instrumented=self.manager.mode == 'on',
                    original_native_lookup_and_bounds_check=True, immutable_norm_plus_one_cached=True)


class NativeLookupManager:
    def __init__(self, runner, mode):
        if mode not in ('off', 'validate', 'on'):
            raise ValueError('Native lookup cache requires off,validate,on')
        self.runner, self.mode, self.helpers = runner, mode, {}
        self.target_order, self.frame = (), None

    def bind_model(self):
        if self.mode == 'off':
            return
        runner, cfg = self.runner, self.runner.vllm_config
        parallel, model = cfg.parallel_config, cfg.model_config
        if (getattr(model.hf_text_config, 'model_type', '') not in ('qwen3_5_moe', 'qwen3_5_moe_text')
                or model.dtype != torch.bfloat16 or model.quantization is not None
                or cfg.lora_config is not None or cfg.kv_transfer_config is not None
                or cfg.speculative_config is None or cfg.speculative_config.method != 'mtp'
                or getattr(model, 'enable_sleep_mode', False)
                or parallel.tensor_parallel_size != 2 or parallel.data_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
                or runner._q35_gdn_input_manager.mode != self.mode):
            raise RuntimeError('Native lookup reuse requires immutable BF16 Q35 TP2/DP1/PP1 input scope')
        for role, module in (('target', runner.model), ('draft', runner.drafter.model)):
            for _, attention in module.named_modules():
                if not all(hasattr(attention, name) for name in ('qkv_proj', 'q_norm', 'k_norm', 'rotary_emb', 'attn')):
                    continue
                helper = LookupHelper(self, attention, role)
                if helper.prefix in self.helpers:
                    raise RuntimeError('Duplicate native attention lookup prefix')
                if (helper.cache.ndim != 2 or helper.cache.shape[1] != 64
                        or helper.cache.shape[0] < model.max_model_len
                        or helper.cache.device != helper.q_source.device
                        or helper.cache.dtype not in (torch.float32, torch.bfloat16)
                        or helper.q_source.dtype not in (torch.bfloat16, torch.float32)
                        or helper.k_source.dtype != helper.q_source.dtype):
                    raise RuntimeError('Pinned native rotary or normalization geometry changed')
                attention._q35_lookup_cache = attention.attn._q35_lookup_cache = helper
                self.helpers[helper.prefix] = helper
        targets = [helper for helper in self.helpers.values() if helper.role == 'target']
        targets.sort(key=lambda helper: int(re.search(r'layers\.(\d+)', helper.prefix).group(1)))
        indices = [int(re.search(r'layers\.(\d+)', helper.prefix).group(1)) for helper in targets]
        if indices != list(range(3, 40, 4)) or len(self.helpers) != 11:
            raise RuntimeError('Require all ten target FIA layers and one native draft FIA layer')
        if len({tensor_key(helper.cache) for helper in targets}) != 1:
            raise RuntimeError('Target rotary lookups do not share the exact same immutable cache storage')
        self.target_order = tuple(helper.prefix for helper in targets)

    def lookup(self, helper, positions, context):
        if helper.role != 'target' or not self.target_order:
            return helper.native(positions), False
        prefix, key = helper.prefix, tensor_key(positions)
        if prefix == self.target_order[0]:
            self.frame = None
            result = helper.native(positions)
            self.frame = dict(context=context, positions=positions, key=key, next=1, value=result)
            return result, False
        frame = self.frame
        if (frame is None or frame['context'] is not context or frame['key'] != key
                or not 0 <= frame['next'] < len(self.target_order)
                or self.target_order[frame['next']] != prefix):
            self.frame = None
            return helper.native(positions), False
        result = frame['value']
        frame['next'] += 1
        if frame['next'] == len(self.target_order):
            self.frame = None
        return result, True

    def reset(self):
        self.frame = None
        for helper in self.helpers.values():
            helper.reset()

    def attach_metrics(self, records):
        if self.mode != 'off' and records:
            records[0]['native_lookup'] = [helper.metrics() for helper in self.helpers.values()]
        return records


def native_rotary_lookup(positions: torch.Tensor, layer_name: str) -> torch.Tensor:
    context = get_forward_context()
    helper = context.no_compile_layers[layer_name]._q35_lookup_cache
    return helper.lookup(positions)


def native_rotary_lookup_fake(positions: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty((*positions.shape, 64), dtype=torch.bfloat16, device=positions.device)


direct_register_custom_op(op_name='q35_native_rotary_lookup', op_func=native_rotary_lookup,
                          fake_impl=native_rotary_lookup_fake, mutates_args=[], dispatch_key='PrivateUse1')
