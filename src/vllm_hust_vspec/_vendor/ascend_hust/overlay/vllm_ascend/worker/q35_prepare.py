"""Opt-in Q35 preparation: immutable speculative indices and a later CPU wait.

Cached tensors contain geometry only. Draft IDs are gathered from the current
input on every call, and every caller receives a fresh metadata object/list.
The original CPU-count event is still synchronized once before inputs return.
"""

import os
from collections import Counter

import numpy as np
import torch
import triton

from vllm_ascend import envs


def uniform_spec_indices(drafts, scheduled):
    """Return native-equivalent index arrays for a homogeneous decode frame."""
    if (drafts.ndim != 1 or scheduled.ndim != 1 or not len(drafts)
            or len(drafts) != len(scheduled)
            or drafts.dtype != np.int32 or scheduled.dtype not in (np.int32, np.int64)):
        return None
    gamma = int(drafts[0])
    n = len(drafts)
    if gamma not in (2, 4, 6, 8, 10, 12) or np.any(drafts != gamma):
        return None
    cu_sampled = np.arange(1, n + 1, dtype=np.int32) * (gamma + 1)
    if not np.array_equal(scheduled, cu_sampled):
        return None
    logits = np.arange(n * (gamma + 1), dtype=scheduled.dtype)
    target = logits.reshape(n, gamma + 1)[:, :gamma].reshape(-1).astype(np.int32)
    return {
        "cu_num_draft_tokens": np.arange(1, n + 1, dtype=np.int32) * gamma,
        "cu_num_sampled_tokens": cu_sampled,
        "logits_indices": logits,
        "target_logits_indices": target,
        "bonus_logits_indices": cu_sampled - 1,
        "draft_input_indices": logits[target + 1],
    }


class Q35PrepareOptimizations:
    def __init__(self, runner, mode, metadata_type):
        if mode not in ("off", "validate", "on"):
            raise ValueError("Q35 preparation mode must be off, validate, or on")
        self.mode = mode
        runtime_setting = envs.VLLM_ASCEND_Q35_RUNTIME_POSITION_LENGTH
        if runtime_setting not in ('0', '1'):
            raise ValueError('VLLM_ASCEND_Q35_RUNTIME_POSITION_LENGTH must be0or1')
        self.runtime_position_length = runtime_setting == '1'
        self.runner = runner
        self.metadata_type = metadata_type
        self.cache = {}
        self.counts = Counter()
        self.validation_counts = Counter()
        self.position_validation_counts = Counter()
        self.enabled = (
            mode != "off" and runner.use_async_spec_decode and not runner.use_dcp
            and runner.lora_config is None and not runner.use_compress
            and getattr(runner.speculative_config, "method", None) == "mtp"
            and getattr(runner.model_config.hf_text_config, "model_type", "")
            in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text")
        )
        if mode != 'off':
            print(f'[q35-prepare-init] pid={os.getpid()} mode={mode} enabled={self.enabled}', flush=True)

    def build(self, drafts, scheduled):
        if not self.enabled or len(drafts) > self.runner.max_num_reqs:
            return None
        arrays = uniform_spec_indices(drafts, scheduled)
        if arrays is None:
            return None
        key = (len(drafts), int(drafts[0]), scheduled.dtype.str)
        indices = self.cache.get(key)
        if indices is None:
            # Bounded to four layouts per request count (two gammas and two
            # native cumulative-index integer widths). These immutable
            # tensors remain alive for all asynchronous readers.
            indices = {
                name: torch.from_numpy(value).pin_memory().to(
                    self.runner.device, non_blocking=True
                ) for name, value in arrays.items()
            }
            self.cache[key] = indices
            self.counts["misses"] += 1
        else:
            self.counts["hits"] += 1
        self.counts[key[1]] += 1
        result = self.metadata_type(
            draft_token_ids=self.runner.input_ids.gpu[indices["draft_input_indices"]],
            num_draft_tokens=drafts.tolist(),
            **{name: tensor for name, tensor in indices.items() if name != "draft_input_indices"},
        )
        total = sum(self.counts[gamma] for gamma in (2, 4, 6, 8, 10, 12))
        if total in (1, 2, 128, 1024):
            print(f"[q35-prepare] pid={os.getpid()} mode={self.mode} counts={dict(self.counts)}", flush=True)
        return result

    def build_positions(self, requests, query_positions, output):
        """Avoid intermediate gathers/casts and the faulting native int64 Add.

        Validation compares exact signed-integer results on CPU; the on path
        neither synchronizes nor reads device values. Other models keep the
        original runner implementation through the existing admission gate.
        """
        if not self.enabled:
            return False
        if (output.device.type != 'npu'
                or any(tensor.device != output.device
                       for tensor in (requests, query_positions, self.runner.num_computed_tokens))
                or output.dtype != torch.int64 or output.ndim != 1
                or requests.dtype not in (torch.int32, torch.int64)
                or query_positions.dtype not in (torch.int32, torch.int64)
                or requests.ndim != 1 or query_positions.ndim != 1
                or requests.numel() != output.numel() or query_positions.numel() != output.numel()
                or any(tensor.stride(0) != 1 for tensor in (requests, query_positions, output))):
            return False
        # Lazy import keeps CPU-only metadata tests independent of NPU code.
        from vllm_ascend.ops.triton.q35_positions import exact_target_positions

        if self.runtime_position_length:
            from vllm_ascend.ops.triton.q35_positions import runtime_target_positions

            exact_target_positions = runtime_target_positions

        tokens = output.numel()
        exact_target_positions[(triton.cdiv(tokens, 256),)](
            self.runner.num_computed_tokens, requests, query_positions, output,
            TOKENS=tokens, BLOCK=256,
        )
        self.position_validation_counts[tokens] += 1
        checks = self.position_validation_counts[tokens]
        if self.mode == 'validate' and checks in (1, 2, 128):
            expected = (self.runner.num_computed_tokens.cpu()[requests.cpu().long()].long()
                        + query_positions.cpu().long())
            assert torch.equal(output.cpu(), expected), 'Q35 integer positions differ from CPU reference'
            print(f'[q35-position-parity] pid={os.getpid()} tokens={tokens} checks={checks}', flush=True)
        return True

    def validate(self, cached, native):
        key = (len(native.num_draft_tokens), native.max_spec_len)
        self.validation_counts[key] += 1
        if self.validation_counts[key] not in (1, 2, 128):
            return
        assert cached.num_draft_tokens == native.num_draft_tokens
        assert cached.max_spec_len == native.max_spec_len
        for name in ("draft_token_ids", "cu_num_draft_tokens", "cu_num_sampled_tokens",
                     "target_logits_indices", "bonus_logits_indices", "logits_indices"):
            actual, expected = getattr(cached, name), getattr(native, name)
            assert (actual.shape, actual.dtype, actual.device) == (
                expected.shape, expected.dtype, expected.device), name
            assert torch.equal(actual, expected), name
        print(f"[q35-prepare-parity] pid={os.getpid()} n={key[0]} gamma={key[1]} "
              f"checks={self.validation_counts[key]}", flush=True)
