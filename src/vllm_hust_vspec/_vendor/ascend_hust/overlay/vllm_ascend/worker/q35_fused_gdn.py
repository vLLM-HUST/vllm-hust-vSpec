"""Private, opt-in pure-speculation GDN metadata refresh.

The runner defers exactly three independent GDN groups until their common
metadata is ready. Admission is CPU-only and happens before any device write.
The launch preserves each builder's captured storage; it does not share state
tables, change accepted-token counts, or alter native Mamba preprocessing.
"""

from collections import Counter
from dataclasses import fields, is_dataclass
import os
import time

import numpy as np
import torch
import triton

from vllm_ascend import envs
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionMetadataBuilder,
    GDNAttentionMetadata,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)
from vllm_ascend.ops.triton.q35_gdn_fused_refresh import gdn_gid123_full_stable_refresh

BUFFER_NAMES = (
    "spec_state_indices_tensor", "spec_sequence_masks", "spec_token_indx",
    "spec_query_start_loc", "num_accepted_tokens", "spec_actual_seq_lengths",
    "non_spec_token_indx",
)
GDN_GROUP_IDS = (1, 2, 3)
STATE_WIDTH = 7
KERNEL_BLOCK = 128


def _precompile_tail_mamba_kernels(runner):
    """Compile cold-path Mamba variants without a model step or device write.

    The 14-request short pilot does not reach the single-request tail or the
    block-copy path of a long formal run. Both are compile-time specializations
    on this Qwen3.5/Ascend stack; putting their compilation in the pilot avoids
    counting compiler work as generation throughput.
    """
    from vllm.v1.worker.mamba_utils import _TEMPORAL_TILES, is_conv_state_dim_first
    from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
    from vllm_ascend.ops.triton.mamba.postprocess import postprocess_mamba_fused_kernel

    buffers = runner._get_mamba_bufs()
    context = buffers.postprocess_align
    if context is None or not context.is_initialized or not context.block_table_stride_req:
        return False
    postprocess_mamba_fused_kernel.warmup(
        torch.int32, torch.int32, torch.int32, torch.int32, torch.int32,
        context.block_table_ptrs, context.block_table_stride_req,
        context.state_base_addrs, context.state_block_strides,
        context.state_elem_sizes, context.state_inner_sizes,
        context.state_conv_widths, context.state_group_indices,
        context.state_dim_row_count, context.state_dim_row_stride,
        context.num_accepted_tokens_out, None, 1,
        grid=(1, 1, _TEMPORAL_TILES), block_size=context.block_size,
        COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
        TEMPORAL_TILES=_TEMPORAL_TILES,
    )
    batch_memcpy_kernel.warmup(
        buffers.preprocess.src_ptrs.gpu, buffers.preprocess.dst_ptrs.gpu,
        buffers.preprocess.sizes.gpu,
        grid=(1,), BLOCK_SIZE=8192,
    )
    return True


def pure_spec_gamma(prefilling, drafts, query, upper, tokens):
    """CPU-only frame gate. Returns gamma or None; no device synchronization."""
    n = len(drafts)
    if not n or len(prefilling) != n or len(query) != n + 1 or len(upper) != n:
        return None
    gamma = int(drafts[0])
    if gamma not in (2, 4, 6, 8, 10, 12) or np.any(drafts != gamma) or np.any(prefilling):
        return None
    lengths = np.diff(query)
    if query[0] != 0 or query[-1] != tokens:
        return None
    if np.any(lengths < 1) or np.any(lengths > gamma + 1) or np.any(upper < lengths):
        return None
    return gamma


def _metadata(builder, n, tokens):
    result = GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=n, num_spec_decode_tokens=tokens, num_actual_tokens=tokens,
        spec_state_indices_tensor=builder.spec_state_indices_tensor[:n],
        spec_sequence_masks=builder.spec_sequence_masks[:n],
        spec_token_indx=builder.spec_token_indx[:tokens],
        non_spec_token_indx=builder.non_spec_token_indx[:0],
        spec_query_start_loc=builder.spec_query_start_loc[:n + 1],
        num_accepted_tokens=builder.num_accepted_tokens[:n],
    )
    result.non_spec_prefill_metadata = None
    result.non_spec_decode_metadata = None
    result.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
            query_start_loc=result.spec_query_start_loc,
            cache_indices=result.spec_state_indices_tensor,
            num_accepted_tokens=result.num_accepted_tokens,
        ),
        actual_seq_lengths=builder.spec_actual_seq_lengths[:n + 1],
    )
    return result


def _same_metadata(a, b, path="metadata"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert (a.shape, a.stride(), a.dtype, a.device, a.data_ptr()) == (
            b.shape, b.stride(), b.dtype, b.device, b.data_ptr()), path
    elif is_dataclass(a):
        assert type(a) is type(b), path
        # Include Ascend attachments, which are not declared upstream fields.
        for name in set(vars(a)) | {f.name for f in fields(a)}:
            _same_metadata(getattr(a, name), getattr(b, name), f"{path}.{name}")
    else:
        assert a == b, path


class FusedGDNRefresh:
    def __init__(self, runner):
        self.mode = envs.VLLM_ASCEND_Q35_GDN_FUSED_REFRESH
        if self.mode not in ("off", "validate", "on"):
            raise ValueError("Q35 GDN refresh must be off, validate, or on")
        self.runner = runner
        self.counts = Counter()
        # Adaptive _runner_query_width owns distinct state buffers for q5/q7;
        # masks/tokens/query/accepted buffers remain common within each group.
        self.storage_signatures = {}
        # Only scalar geometry and views of persistent builder-owned tensors
        # live in these metadata objects. Values are refreshed by the kernel.
        # Bound: two layouts * max requests * capture-token capacity.
        self.metadata_cache = {}
        self.last_gamma = None
        self.precompiled_geometries = set()
        self.precompiled_tail_mamba = False
        if self.mode != "off":
            print(f"[q35-gdn-fused] pid={os.getpid()} initialized mode={self.mode}", flush=True)

    def eligible(self, *, full, capture, ubatches, spec, n, padded_n, tokens, padded_tokens):
        r = self.runner
        return (
            self.mode != "off" and full and not capture and ubatches is None
            # FIA can pad tokens while retaining the same request count. GDN
            # consumes its separate *unpadded* query locations and actual
            # token count. A padded request is still rejected here.
            and spec and n == padded_n and 0 < tokens <= padded_tokens
            and r.use_async_spec_decode and not r.use_dcp
            and r.cache_config.mamba_cache_mode == "align"
            and r.speculative_config.num_speculative_tokens in (4, 6, 8, 10, 12)
            and getattr(r.model_config.hf_text_config, "model_type", "")
            in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text")
            and len(r.attn_groups) == 4 and all(len(g) == 1 for g in r.attn_groups)
        )

    def _metadata_for(self, builders, width, n, tokens):
        key = (width, n, tokens)
        results = self.metadata_cache.get(key)
        if results is None:
            results = tuple(_metadata(b, n, tokens) for b in builders)
            self.metadata_cache[key] = results
        else:
            self.counts["metadata_cache_hits"] += 1
        return results

    def build(self, pending):
        if tuple(gid for gid, _ in pending) != GDN_GROUP_IDS:
            return None
        r = self.runner
        builders = [r.attn_groups[gid][0].get_metadata_builder(0) for gid, _ in pending]
        if any(type(b) is not AscendGDNAttentionMetadataBuilder for b in builders):
            return None
        commons = [cm for _, cm in pending]
        cm = commons[0]
        n, tokens = cm.num_reqs, cm.num_actual_tokens
        upper = cm.seq_lens_cpu_upper_bound.numpy()
        gamma = pure_spec_gamma(cm.is_prefilling.numpy(),
                                r.num_decode_draft_tokens.cpu[:n].numpy(),
                                cm.query_start_loc_cpu.numpy(), upper, tokens)
        if gamma is None:
            return None
        tables = [m.block_table_tensor for m in commons]
        groups = [[getattr(b, name) for name in BUFFER_NAMES] for b in builders]
        capacity, width = groups[0][0].shape
        block_size = builders[0].kv_cache_spec.block_size
        if width not in (3, 5, 7, 9, 11, 13) or width < gamma + 1 or n > capacity or tokens > capacity:
            return None
        for b, m, table, buffers in zip(builders, commons, tables, groups):
            if (not b.use_full_cuda_graph or not b.use_spec_decode or b.num_spec != width - 1
                    or tokens > b.decode_cudagraph_max_bs
                    or b.kv_cache_spec.block_size != block_size
                    or b.kv_cache_spec.num_speculative_blocks < width - 1
                    or table.shape != tables[0].shape or table.stride() != tables[0].stride()
                    or table.shape[0] != n or table.stride(1) != 1
                    or int(upper.max() - 1) // block_size + width > table.shape[1]
                    or m.seq_lens.data_ptr() != cm.seq_lens.data_ptr()
                    or m.query_start_loc.data_ptr() != cm.query_start_loc.data_ptr()):
                return None
            if any(x.shape != y.shape or x.dtype != y.dtype or not x.is_contiguous()
                   or x.device != table.device for x, y in zip(buffers, groups[0])):
                return None
        inputs = [*tables, cm.seq_lens, cm.query_start_loc, r.num_accepted_tokens.gpu[:n]]
        if any(t.dtype != torch.int32 or t.device != tables[0].device for t in inputs):
            return None
        if any(not t.is_contiguous() for t in inputs[3:]):
            return None
        signature = tuple((id(b), tuple((x.data_ptr(), tuple(x.shape), x.dtype)
                                       for x in buffers)) for b, buffers in zip(builders, groups))
        if width not in self.storage_signatures:
            addresses = [x.untyped_storage().data_ptr() for buffers in groups for x in buffers]
            assert len(set(addresses)) == len(addresses), "GDN group buffers alias"
            self.storage_signatures[width] = signature
        else:
            assert signature == self.storage_signatures[width], "Captured GDN storage changed"
        # Metadata construction can fail only before any launch; never fallback
        # after a partially completed refresh.
        results = self._metadata_for(builders, width, n, tokens)
        kernel_inputs = (*inputs, *(x for buffers in groups for x in buffers[:6]), n, tokens)
        kernel_constants = dict(
            TABLE_COLS=tables[0].shape[1], TABLE_STRIDE0=tables[0].stride(0),
            STATE_CAPACITY=capacity, STATE_WIDTH=width, MAMBA_BLOCK_SIZE=block_size,
            BLOCK=KERNEL_BLOCK,
        )
        # A short pilot can exercise only the starting gamma.  Compiling a
        # previously unseen width during the timed workload blocks the worker
        # for several seconds.  Triton warmup compiles without launching or
        # touching model state, so prepare all configured even-gamma widths on
        # the first admitted pilot frame when explicitly requested.
        if os.environ.get("VLLM_ASCEND_Q35_GDN_PRECOMPILE_WIDTHS") == "1":
            geometry = (kernel_constants["TABLE_COLS"], kernel_constants["TABLE_STRIDE0"],
                        capacity, block_size)
            if geometry not in self.precompiled_geometries:
                started = time.monotonic()
                max_gamma = int(r.speculative_config.num_speculative_tokens)
                widths = tuple(range(3, max_gamma + 2, 2))
                for candidate_width in widths:
                    candidate_constants = dict(kernel_constants, STATE_WIDTH=candidate_width)
                    gdn_gid123_full_stable_refresh.warmup(
                        *kernel_inputs,
                        grid=(triton.cdiv(capacity * candidate_width, KERNEL_BLOCK),),
                        **candidate_constants,
                    )
                self.precompiled_geometries.add(geometry)
                print(f"[q35-gdn-fused] pid={os.getpid()} precompiled_widths={widths} "
                      f"seconds={time.monotonic() - started:.3f}", flush=True)
            if not self.precompiled_tail_mamba:
                started = time.monotonic()
                self.precompiled_tail_mamba = _precompile_tail_mamba_kernels(r)
                if self.precompiled_tail_mamba:
                    print(f"[q35-gdn-fused] pid={os.getpid()} precompiled_tail_mamba "
                          f"seconds={time.monotonic() - started:.3f}", flush=True)
        gdn_gid123_full_stable_refresh[(triton.cdiv(capacity * width, KERNEL_BLOCK),)](
            *kernel_inputs, **kernel_constants,
        )
        if self.mode == "validate":
            snapshots = [[x.clone() for x in buffers] for buffers in groups]
            native_results = []
            for b, m, candidate, buffers, snapshot in zip(builders, commons, results, groups, snapshots):
                native = b.build(common_prefix_len=0, common_attn_metadata=m,
                                 num_accepted_tokens=inputs[-1],
                                 num_decode_draft_tokens_cpu=r.num_decode_draft_tokens.cpu[:n])
                _same_metadata(candidate, native)
                for name, fused, actual in zip(BUFFER_NAMES, snapshot, buffers):
                    assert torch.equal(fused, actual), f"GDN full buffer mismatch: {name}"
                native_results.append(native)
            results = native_results
        self.counts[gamma] += 1
        self.counts[f"width{width}"] += 1
        self.counts["padded_token_frames"] += int(cm.num_input_tokens > tokens)
        if self.last_gamma is not None and self.last_gamma != gamma:
            self.counts["transitions"] += 1
        self.last_gamma = gamma
        total = sum(self.counts[value] for value in (2, 4, 6, 8, 10, 12))
        if total == 1 or total % 128 == 0 or self.counts[gamma] == 1:
            print(f"[q35-gdn-fused] pid={os.getpid()} mode={self.mode} "
                  f"counts={dict(self.counts)} n={n} tokens={tokens} "
                  f"input_tokens={cm.num_input_tokens} width={width}", flush=True)
        return results
