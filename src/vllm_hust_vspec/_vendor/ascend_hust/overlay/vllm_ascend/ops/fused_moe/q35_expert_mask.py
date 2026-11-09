"""Native top8 mask fusion with captured exact native-shadow flags."""

import os

import torch

from vllm_ascend import envs


class Q35ExpertMask:
    def __init__(self, top_k, num_local_experts):
        self.mode = envs.VLLM_ASCEND_Q35_EXPERT_MASK
        if self.mode not in ('off', 'validate', 'on'):
            raise ValueError('Expert mask mode must be off, validate, or on')
        self.enabled = self.mode != 'off'
        self.counts = self.errors = None
        self.captured = set()
        if not self.enabled:
            return
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        if (top_k != 8 or num_local_experts != 128
                or getattr(config.model_config.hf_text_config, 'model_type', '')
                not in ('qwen3_5_moe', 'qwen3_5_moe_text')
                or config.model_config.dtype != torch.bfloat16
                or config.model_config.quantization is not None or config.lora_config is not None
                or config.parallel_config.tensor_parallel_size != 2
                or config.parallel_config.data_parallel_size != 1):
            raise RuntimeError('Private expert mask requires BF16 Q35 TP2 top8/local128')

    def apply(self, ids, weights, expert_map):
        if (not self.enabled or ids.ndim != 2 or ids.shape[1] != 8 or ids.shape[0] > 208
                or expert_map.numel() != 256 or expert_map.dtype != torch.int32
                or ids.dtype not in (torch.int32, torch.int64)
                or weights.dtype not in (torch.bfloat16, torch.float32)
                or not all(value.is_contiguous() for value in (ids, weights, expert_map))):
            return None
        from vllm_ascend.ops.triton.q35_expert_mask import fused_mask

        tokens = ids.shape[0]
        if self.counts is None:
            self.counts = torch.zeros(209, dtype=torch.int64, device=weights.device)
            self.errors = torch.zeros(209, dtype=torch.int32, device=weights.device)
        actual = fused_mask(ids, weights, expert_map)
        self.counts[tokens].add_(1)
        if self.mode == 'validate':
            expected = weights * (expert_map[ids] != -1)
            view = torch.int16 if weights.dtype == torch.bfloat16 else torch.int32
            flag = (actual.view(view) != expected.view(view)).any().to(torch.int32)
            torch.maximum(self.errors[tokens], flag, out=self.errors[tokens])
        if tokens not in self.captured:
            self.captured.add(tokens)
            print(f'[q35-expert-mask] pid={os.getpid()} mode={self.mode} tokens={tokens} '
                  'native_top8_producer=True native_expert_map_read=True', flush=True)
        return actual

    @torch.inference_mode()
    def reset(self):
        if self.counts is not None:
            self.counts.zero_()
            self.errors.zero_()

    def metrics(self):
        if self.counts is None:
            return dict(enabled=False, mode=self.mode)
        counts, errors = self.counts.cpu().tolist(), self.errors.cpu().tolist()
        return dict(enabled=True, mode=self.mode, captured_shapes=sorted(self.captured),
            actual_graph_counts={str(t): count for t, count in enumerate(counts) if count},
            native_bitwise_error_flags={str(t): errors[t] for t, count in enumerate(counts) if count})
