"""Opt-in exact device-masked target attention with cached graph tiling.

The native CPU correction wait is retained. The only cached host parameters
are conservative rounded bounds and grouped query shapes; exact current
causality, physical pages and lengths are rebuilt from device inputs each
frame. Strict speculative rejection and target logits are unchanged.
"""

import os
from collections import Counter
from dataclasses import dataclass

import torch
import torch_npu
import triton
from vllm.config import CUDAGraphMode

from vllm_ascend.ops.triton.q35_masked_fia import (
    exact_mask_geometry,
    grouped_query_pack,
    grouped_query_unpack,
    zero_invisible_tails,
)
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.q35_masked_fia_checks import attention_comparison
from vllm_ascend.worker.q35_masked_fia_geometry import MAX_WIDTH, PAGE_SIZE, attention_graph_key, masked_layout

KERNEL_BLOCK = 1024


@dataclass
class MaskedFIAEntry:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    packed: torch.Tensor
    grouped_output: torch.Tensor
    lse: torch.Tensor
    workspace: torch.Tensor
    scale: float
    event: object
    handle: object
    last_layout: object
    native_mask: torch.Tensor | None = None
    native_output: torch.Tensor | None = None
    native_lse: torch.Tensor | None = None
    parity: torch.Tensor | None = None
    diagnostics: torch.Tensor | None = None


class Q35MaskedFIA:
    def __init__(self, runner, mode):
        if mode not in ("off", "validate", "on"):
            raise ValueError("Masked FIA mode must be off, validate, or on")
        self.runner, self.mode = runner, mode
        self.enabled = (
            mode != "off"
            and runner.use_async_spec_decode
            and not runner.use_dcp
            and runner.lora_config is None
            and not runner.use_compress
            and getattr(runner.speculative_config, "method", None) == "mtp"
            and getattr(runner.model_config.hf_text_config, "model_type", "")
            in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text")
            and runner.max_num_reqs <= 16
            and runner.max_model_len % PAGE_SIZE == 0
        )
        self.captures, self.counts = {}, Counter()
        self.workspace = self.packed = self.grouped_output = self.native_output = None
        self.control = self.mask = self.table = None
        self.layout = self.metadata = self.device_seq = self.qsl = None
        self.nreqs = 0
        if mode != "off":
            print(f"[q35-masked-fia-init] pid={os.getpid()} mode={mode} enabled={self.enabled}", flush=True)

    def diagnostic_boundary(self, stage):
        """Bounded synchronization only in the explicitly diagnostic mode.

        ACL graphs prohibit global launch blocking. These first-three-frame
        boundaries distinguish prior draft failures from preparation failures;
        they are absent from hot-mode device execution.
        """
        if not self.enabled or self.mode != "validate":
            return
        key = f"diagnostic_{stage}"
        self.counts[key] += 1
        if self.counts[key] > 3:
            return
        print(f"[q35-masked-boundary] pid={os.getpid()} stage={stage} begin={self.counts[key]}", flush=True)
        torch.npu.synchronize()
        print(f"[q35-masked-boundary] pid={os.getpid()} stage={stage} passed={self.counts[key]}", flush=True)

    def prepare(self, metadata, common, actual_reqs, capture):
        if not self.enabled or not hasattr(metadata, "seq_lens_list"):
            return
        max_tokens = max(self.runner.vllm_config.compilation_config.cudagraph_capture_sizes)
        tokens = metadata.actual_seq_lengths_q[-1]
        if tokens > max_tokens:
            return  # Larger prefill frames execute the native eager path.
        upper = metadata.seq_lens_list
        # Dummy capture metadata may have zero device lengths. Host capture
        # bounds must still cover every structural query and all future replays.
        if capture:
            upper = [self.runner.max_model_len] * len(upper)
        layout = masked_layout(metadata.actual_seq_lengths_q, upper, actual_reqs, self.runner.max_model_len)
        assert metadata.causal and metadata.block_tables.shape[1] >= self.runner.max_model_len // PAGE_SIZE
        assert common.seq_lens.device.type == "npu" and common.seq_lens.numel() >= actual_reqs
        if self.control is None:
            self.row_capacity = max_tokens + MAX_WIDTH
            self.columns = self.runner.max_model_len // PAGE_SIZE
            self.control = torch.empty(4, dtype=torch.int32, device=self.runner.device)
            self.mask = torch.empty(
                (self.row_capacity, self.runner.max_model_len), dtype=torch.bool, device=self.runner.device
            )
            self.table = torch.empty((self.row_capacity, self.columns), dtype=torch.int32, device=self.runner.device)
            # Target full graphs execute serially on the runner's compute
            # stream. Pack -> FIA -> unpack (and shadow comparison) completes
            # before the next layer may overwrite this runner-owned arena.
            self.packed = torch.empty(self.row_capacity * 8 * 256, dtype=torch.bfloat16, device=self.runner.device)
            self.grouped_output = torch.empty_like(self.packed)
            if self.mode == "validate":
                self.native_output = torch.empty((max_tokens, 8, 256), dtype=torch.bfloat16, device=self.runner.device)
        self.layout, self.metadata, self.device_seq = layout, metadata, common.seq_lens
        self.qsl, self.nreqs = metadata.query_start_loc, actual_reqs
        # Fresh pinned source remains owned by the asynchronous copy operation.
        actual_tokens = metadata.actual_seq_lengths_q[actual_reqs - 1]
        control = torch.tensor([layout.width, tokens, actual_reqs, actual_tokens],
                               dtype=torch.int32, pin_memory=True)
        self.control.copy_(control, non_blocking=True)
        exact_mask_geometry[(self.row_capacity, triton.cdiv(self.runner.max_model_len, KERNEL_BLOCK))](
            self.device_seq,
            self.qsl,
            metadata.block_tables,
            self.control,
            self.mask,
            self.table,
            MAX_CONTEXT=self.runner.max_model_len,
            TABLE_COLS=self.columns,
            TABLE_STRIDE=metadata.block_tables.stride(0),
            MAX_REQS=triton.next_power_of_2(self.runner.max_num_reqs),
            ROW_CAPACITY=self.row_capacity,
            PAGE=PAGE_SIZE,
            BLOCK=KERNEL_BLOCK,
        )
        metadata.q35_masked_fia = self

    def _parameters(self, entry, layout):
        groups, width, heads, dim = layout.groups, layout.width, entry.query.shape[1], entry.query.shape[2]
        count = groups * heads * width * dim
        return dict(
            query=entry.packed[:count].view(groups, heads, width, dim),
            key=entry.key,
            value=entry.value,
            block_table=self.table[:groups],
            atten_mask=self.mask[: groups * width].view(groups, 1, width, self.runner.max_model_len),
            input_layout="BNSD",
            block_size=PAGE_SIZE,
            actual_seq_lengths=[width] * groups,
            actual_seq_lengths_kv=list(layout.upper_lengths),
            num_key_value_heads=1,
            num_heads=heads,
            scale=entry.scale,
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )

    def _call(self, entry, layout):
        heads, dim = entry.query.shape[1:]
        count = layout.groups * heads * layout.width * dim
        torch_npu.npu_fused_infer_attention_score.out(
            **self._parameters(entry, layout),
            workspace=entry.workspace,
            out=[entry.grouped_output[:count].view(layout.groups, heads, layout.width, dim), entry.lse],
        )
        if entry.native_output is not None:
            metadata = self.metadata
            torch_npu.npu_fused_infer_attention_score.out(
                query=entry.query,
                key=entry.key,
                value=entry.value,
                block_table=metadata.block_tables,
                atten_mask=entry.native_mask,
                input_layout="TND",
                block_size=PAGE_SIZE,
                actual_seq_lengths=metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=metadata.seq_lens_list,
                num_key_value_heads=1,
                num_heads=heads,
                scale=entry.scale,
                sparse_mode=3,
                pre_tokens=2147483647,
                next_tokens=2147483647,
                workspace=entry.workspace,
                out=[entry.native_output, entry.native_lse],
            )

    def has_capture(self, context, tokens):
        return attention_graph_key(context, tokens) in self.captures

    def capture(self, backend, query, key, value, metadata, output, context):
        assert backend.num_kv_heads == 1 and backend.num_heads == 8 and backend.head_size == 256
        assert not backend.sliding_window and not backend.enable_c8_quant
        key, value, page, _, _ = backend._get_fia_params(key, value, metadata)
        assert page == PAGE_SIZE and query.dtype == torch.bfloat16
        assert key.is_contiguous() and value.is_contiguous(), 'Raw KV sanitization requires dense planes'
        assert torch_npu.get_npu_format(key) in (0, 2) and torch_npu.get_npu_format(value) in (0, 2), \
            'Raw KV sanitization does not support packed KV formats'
        layout = self.layout
        tokens, heads, dim = layout.tokens, backend.num_heads, backend.head_size
        query = query[:tokens].view(tokens, heads, dim)
        output = output[:tokens].view(tokens, heads, dim)
        assert query.is_contiguous() and output.is_contiguous()
        capacity = (tokens + MAX_WIDTH) * heads * dim
        packed = self.packed
        entry = MaskedFIAEntry(
            weak_ref_tensors(query),
            weak_ref_tensors(key),
            weak_ref_tensors(value),
            weak_ref_tensors(output),
            packed,
            self.grouped_output,
            torch.empty(1, dtype=query.dtype, device=query.device),
            None,
            backend.scale,
            torch.npu.ExternalEvent(),
            None,
            None,
        )
        if self.mode == "validate":
            entry.native_mask = weak_ref_tensors(metadata.attn_mask)
            entry.native_output = self.native_output[:tokens]
            entry.native_lse = torch.empty_like(entry.lse)
            entry.parity = torch.empty((), dtype=torch.bool, device=query.device)
            entry.diagnostics = torch.empty(5, dtype=torch.float32, device=query.device)
        entry.workspace = self.workspace
        if entry.workspace is None:
            entry.workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                **self._parameters(entry, layout)
            )
        if self.mode == "validate" and self.workspace is None:
            native_workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                block_table=metadata.block_tables,
                atten_mask=metadata.attn_mask,
                input_layout="TND",
                block_size=PAGE_SIZE,
                actual_seq_lengths=metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=metadata.seq_lens_list,
                num_key_value_heads=1,
                num_heads=heads,
                scale=backend.scale,
                sparse_mode=3,
                pre_tokens=2147483647,
                next_tokens=2147483647,
            )
            if native_workspace.numel() > entry.workspace.numel():
                entry.workspace = native_workspace
        if self.workspace is None:
            # CANN's max workspace is shape-independent within the bounded
            # BF16/head256/page128 contract (both-device size screen). Never
            # retain one323MiB workspace per graph key. The same compute
            # stream orders all accesses, including native shadow calls.
            self.workspace = entry.workspace
            print(
                f"[q35-masked-fia-arena] pid={os.getpid()} workspace_bytes="
                f"{self.workspace.numel() * self.workspace.element_size()} "
                f"pack_elements={self.packed.numel()} key_shape={tuple(key.shape)} "
                f"key_stride={key.stride()} key_format={torch_npu.get_npu_format(key)}",
                flush=True,
            )
        zero_invisible_tails[(self.runner.max_num_reqs + 1, triton.cdiv(PAGE_SIZE * dim, KERNEL_BLOCK))](
            key,
            value,
            self.runner.seq_lens,
            self.runner.input_batch.block_table[0].get_device_tensor(),
            self.control,
            PAGE=PAGE_SIZE,
            PLANE_DIM=dim,
            TABLE_STRIDE=self.runner.input_batch.block_table[0].get_device_tensor().stride(0),
            BLOCK=KERNEL_BLOCK,
        )
        grouped_query_pack[(triton.cdiv(capacity, KERNEL_BLOCK),)](
            query, packed, self.control, TOKENS=tokens, HEADS=heads, DIM=dim, CAPACITY=capacity, BLOCK=KERNEL_BLOCK
        )
        stream = torch.npu.current_stream()
        entry.event.wait(stream)
        entry.event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        self._call(entry, layout)
        entry.handle = torch.npu.graph_task_group_end(stream)
        grouped_query_unpack[(triton.cdiv(tokens * heads * dim, KERNEL_BLOCK),)](
            entry.grouped_output, output, self.control, TOKENS=tokens, HEADS=heads, DIM=dim, BLOCK=KERNEL_BLOCK
        )
        if entry.parity is not None:
            passed, diagnostics = attention_comparison(output, entry.native_output, self.control[3])
            entry.parity.copy_(passed)
            entry.diagnostics.copy_(diagnostics)
        graph_key = attention_graph_key(context, tokens)
        self.captures.setdefault(graph_key, []).append(entry)
        return output, tokens

    def update(self, stream, tokens, context):
        graph_key = attention_graph_key(context, tokens)
        assert self.layout.tokens == tokens and graph_key in self.captures
        layout = self.layout
        key = (layout.width, layout.upper_lengths)
        with torch.npu.stream(stream):
            for entry in self.captures[graph_key]:
                if self.mode == "validate" or entry.last_layout != key:
                    torch.npu.graph_task_update_begin(stream, entry.handle)
                    self._call(entry, layout)
                    torch.npu.graph_task_update_end(stream)
                    entry.last_layout = key
                    self.counts["retiled_layers"] += 1
                else:
                    self.counts["cached_layers"] += 1
                # Every replay resets its captured event, including cache hits.
                entry.event.record(stream)
        self.counts[layout.width] += 1
        if self.counts[layout.width] in (1, 2, 128, 1024):
            print(f"[q35-masked-fia] pid={os.getpid()} width={layout.width} counts={dict(self.counts)}", flush=True)

    def check(self, context, tokens):
        if (
            self.mode != "validate"
            or context.capturing
            or context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        ):
            return
        graph_key = attention_graph_key(context, tokens)
        if graph_key not in self.captures:
            return
        entries = self.captures[graph_key]
        flags = torch.stack([entry.parity for entry in entries])
        if not bool(flags.all().cpu()):
            diagnostics = torch.stack([entry.diagnostics for entry in entries]).cpu().tolist()
            print(f'[q35-masked-fia-failure] pid={os.getpid()} graph_tokens={tokens} '
                  f'graph_key={graph_key} '
                  f'actual_reqs={self.nreqs} layout={self.layout} '
                  f'query_ends={self.metadata.actual_seq_lengths_q} '
                  f'KV_lengths={self.metadata.seq_lens_list} layer_flags={flags.cpu().tolist()} '
                  f'layer_stats_max_error_valid_nan_candidate_valid_nan_native_pad_nan_candidate_pad_nan_native={diagnostics}',
                  flush=True)
            raise AssertionError('Masked attention differs from native TND or produced non-finite real-token values')
        width = self.layout.width
        self.counts[f"parity_width{width}"] += 1
        if self.counts[f"parity_width{width}"] in (1, 2, 128):
            print(
                f"[q35-masked-fia-parity] pid={os.getpid()} width={width} "
                f"layers={len(entries)} checks={self.counts[f'parity_width{width}']}",
                flush=True,
            )
