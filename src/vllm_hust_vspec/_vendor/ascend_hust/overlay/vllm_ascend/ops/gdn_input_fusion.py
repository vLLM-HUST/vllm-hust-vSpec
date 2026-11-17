"""Private pure-spec Q35 GDN input fusion, with in-graph native-shadow checks.

Per-layer counters/errors are allocated before graph capture. Native shadow
and GPU comparisons are captured in validate mode, so actual graph replays
are checked; no per-step D2H or Python callback is needed. Stats are read by
a named worker RPC after timers. Reset stats after capture, before warmup.
"""

import os

import torch


def eligible(meta, mixed, a, b):
    if meta.spec_sequence_masks is None or meta.num_prefills or meta.num_decodes:
        return False
    n = meta.num_spec_decodes
    tokens = mixed.shape[0] if mixed is not None and mixed.ndim == 2 else 0
    return bool(n > 0 and 0 < tokens <= 208 and tokens in (n * 5, n * 7)
                and mixed.shape[1] == 4096 and mixed.dtype == torch.bfloat16
                and a.shape == b.shape == (tokens, 16)
                and a.dtype == b.dtype == torch.bfloat16
                and all(value.is_contiguous() for value in (mixed, a, b)))


class GDNInputFusion:
    def __init__(self, attention, mode, config=None):
        if mode not in ('off', 'validate', 'on'):
            raise ValueError('GDN input fusion is off,validate,on')
        self.attention, self.mode = attention, mode
        self.enabled = mode != 'off'
        if not self.enabled:
            return
        model = config.model_config
        if (getattr(model.hf_text_config, 'model_type', '') not in ('qwen3_5_moe', 'qwen3_5_moe_text')
                or model.dtype != torch.bfloat16 or getattr(model, 'quantization', None) is not None
                or config.lora_config is not None or config.parallel_config.tensor_parallel_size != 2
                or getattr(config.parallel_config, 'data_parallel_size', 1) != 1
                or attention.head_k_dim != 128 or attention.head_v_dim != 128
                or attention.key_dim // attention.tp_size != 1024
                or attention.value_dim // attention.tp_size != 2048
                or attention.A_log.dtype != torch.float32
                or attention.dt_bias.dtype not in (torch.float32, torch.bfloat16)):
            raise RuntimeError('Private GDN input fusion requires unquantized Q35 BF16 TP2/DP1 geometry')
        self.prefix = attention.prefix
        self.counts = torch.zeros(209, dtype=torch.int64, device=attention.A_log.device)
        self.errors = torch.zeros((209, 5), dtype=torch.int32, device=attention.A_log.device)
        self.max_abs = torch.zeros((209, 5), dtype=torch.float32, device=attention.A_log.device)
        self.captured_tokens = set()

    def apply(self, meta, mixed, a, b):
        if not self.enabled or not eligible(meta, mixed, a, b):
            return None
        if any(value.device != self.attention.A_log.device for value in (mixed, a, b)):
            return None
        from vllm_ascend.ops.triton.q35_fused_gdn_inputs import fused_inputs

        tokens = mixed.shape[0]
        actual = fused_inputs(mixed, a, b, self.attention.A_log, self.attention.dt_bias)
        self.counts[tokens].add_(1)
        if self.mode == 'validate':
            self._native_shadow(mixed, a, b, actual, tokens)
        if tokens not in self.captured_tokens:
            self.captured_tokens.add(tokens)
            print(f'[q35-input-fusion] pid={os.getpid()} layer={self.prefix} '
                  f'mode={self.mode} tokens={tokens} in_graph_counter=True', flush=True)
        return actual

    def _native_shadow(self, mixed, a, b, actual, tokens):
        from vllm_ascend.ops import gdn

        q, k, v = self.attention.rearrange_mixed_qkv(mixed)
        q, k = gdn.l2norm_fwd(q), gdn.l2norm_fwd(k)
        g, beta = gdn.DeviceOperator.fused_gdn_gating(self.attention.A_log, a, b, self.attention.dt_bias)
        reference = (q, k, v, g, beta)
        differences = [(left.float() - right.float()).abs() for left, right in zip(reference, actual)]
        flags = []
        for index, (left, right) in enumerate(zip(reference, actual)):
            mismatch = (((~torch.isfinite(left)) | (~torch.isfinite(right))
                         | (differences[index] > (1e-6 + 1e-6 * left.abs()))).any() if index == 3
                        else (left != right).any())
            flags.append(mismatch.to(torch.int32))
        torch.maximum(self.errors[tokens], torch.stack(flags), out=self.errors[tokens])
        torch.maximum(self.max_abs[tokens], torch.stack([difference.max() for difference in differences]),
                      out=self.max_abs[tokens])

    def reset_stats(self):
        if self.enabled:
            self.counts.zero_()
            self.errors.zero_()
            self.max_abs.zero_()

    def metrics(self):
        if not self.enabled:
            return dict(enabled=False)
        counts, errors, maximum = self.counts.cpu().tolist(), self.errors.cpu().tolist(), self.max_abs.cpu().tolist()
        return dict(enabled=True, mode=self.mode, pid=os.getpid(), layer=self.prefix,
                    dtype_projection='bfloat16', dtype_bias=str(self.attention.dt_bias.dtype),
                    actual_graph_counts={str(index): value for index, value in enumerate(counts) if value},
                    native_shadow_error_flags={str(index): errors[index] for index, value in enumerate(counts) if value},
                    native_shadow_max_abs={str(index): maximum[index] for index, value in enumerate(counts) if value},
                    q_k_v_beta_tolerance=0, fp32_g_atol=1e-6, fp32_g_rtol=1e-6,
                    stats_read_after_timers_only=True)


class GDNInputFusionManager:
    def __init__(self, runner, mode):
        if mode not in ('off', 'validate', 'on'):
            raise ValueError('GDN input fusion is off,validate,on')
        self.runner, self.mode, self.layers = runner, mode, {}

    def bind_model(self, model):
        if self.mode == 'off':
            return
        for _, module in model.named_modules():
            if (hasattr(module, 'rearrange_mixed_qkv') and hasattr(module, 'A_log')
                    and hasattr(module, 'dt_bias') and hasattr(module, 'head_k_dim')):
                helper = GDNInputFusion(module, self.mode, self.runner.vllm_config)
                module._q35_input_fusion = helper
                if helper.prefix in self.layers:
                    raise RuntimeError('Duplicate GDN input fusion layer prefix')
                self.layers[helper.prefix] = helper
        if len(self.layers) != 30:
            raise RuntimeError(f'Private Q35 model must expose30 GDN layers, got{len(self.layers)}')

    def reset_stats(self):
        for helper in self.layers.values():
            helper.reset_stats()
        return dict(mode=self.mode, reset_layers=len(self.layers))

    def metrics(self):
        return [helper.metrics() for helper in self.layers.values()]
