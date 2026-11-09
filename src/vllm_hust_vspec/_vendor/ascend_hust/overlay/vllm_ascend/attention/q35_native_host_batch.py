"""Default-off CPU boxing reduction; original native FIA and events retained."""
from collections import Counter
import importlib.util
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm_hust_vspec.backends.experimental_paths import require_current_npu_proof

PRIMITIVE = Path(os.environ.get('VSPEC_NATIVE_FIA_ARTIFACT_DIR', '/__vspec_artifacts_not_configured__'))


def verify_primitive():
    proof_path = PRIMITIVE / 'primitive-native-audit-r1.json'
    proof = json.loads(proof_path.read_text())
    require_current_npu_proof(proof)
    assert proof['passed']
    assert proof['numerical_comparisons'] == 2160 and proof['scope'] == 'target-draft'
    assert proof['all_consumed_output_rows_finite_and_bitwise_exact']
    assert proof['original_per_group_update_and_ExternalEvent_order_preserved']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in proof['sha256'].items())
    return proof_path


def derive_tasks(attn_keys, params, metadata, draft_steps, is_draft, workspace):
    result = []
    for index, (key, param) in enumerate(zip(attn_keys[:len(params)], params, strict=True)):
        if not isinstance(param, tuple) or len(param) != 22:
            return None
        (query, key_cache, value, table, mask, block_size, old_lengths, old_qs,
         kv_heads, heads, scale, output, lse, sparse, pre, next_tokens, sliding,
         k_scale, k_offset, v_scale, v_offset, layer_name) = param
        if sliding is not None or any(x is not None for x in (k_scale, k_offset, v_scale, v_offset)):
            return None
        if is_draft:
            step, key = draft_steps[index]
            current = metadata[step][key]
            if not current.causal:
                return None
        else:
            current = metadata[layer_name if layer_name is not None and layer_name in metadata else key]
        result.append(dict(query=query, key=key_cache, value=value, block_table=current.block_tables,
            atten_mask=mask, input_layout='TND', block_size=block_size,
            actual_seq_lengths=current.actual_seq_lengths_q, actual_seq_lengths_kv=current.seq_lens_list,
            num_key_value_heads=kv_heads, num_heads=heads, scale=scale, sparse_mode=sparse,
            pre_tokens=pre, next_tokens=next_tokens, workspace=workspace, out=[output, lse]))
    return result


def same_arguments(actual, reference):
    if len(actual) != len(reference):
        raise RuntimeError('Native FIA task cardinality mismatch')
    for candidate, native in zip(actual, reference, strict=True):
        if candidate.keys() != native.keys():
            raise RuntimeError('Native FIA argument names differ')
        for key, value in candidate.items():
            expected = native[key]
            if isinstance(value, torch.Tensor):
                if not isinstance(expected, torch.Tensor) or (value.shape, value.stride(), value.dtype,
                    value.device, value.data_ptr()) != (expected.shape, expected.stride(), expected.dtype,
                                                       expected.device, expected.data_ptr()):
                    raise RuntimeError(f'Native FIA tensor binding mismatch: {key}')
            elif key == 'out':
                for left, right in zip(value, expected, strict=True):
                    if left.data_ptr() != right.data_ptr() or left.shape != right.shape or left.dtype != right.dtype:
                        raise RuntimeError('Native FIA output storage mismatch')
            elif value != expected:
                raise RuntimeError(f'Native FIA host argument mismatch: {key}')


def eligible_rows(rows, role, tokens, requests=14):
    if requests not in (1, 2, 3, 14) or tokens not in (requests * 5, requests * 7):
        return False
    expected = 10 if role == 'target' else tokens // requests - 1
    widths = [tokens // requests] * 10 if role == 'target' else [tokens // requests] + [1] * (expected - 1)
    if rows is None or len(rows) != expected:
        return False
    for index, (row, width) in enumerate(zip(rows, widths, strict=True)):
        q, k, v = row['query'], row['key'], row['value']
        expected_qs = [width * (i + 1) for i in range(requests)]
        if role == 'draft' and index > 0:
            expected_qs = list(range(1, tokens + 1))
        if (q.shape != (tokens, 8, 256) or k.ndim != 3 or k.shape[1:] != (128, 256)
                or v.shape != k.shape or q.dtype != k.dtype or q.dtype != v.dtype
                or q.dtype != torch.bfloat16 or row['out'][0].shape != q.shape
                or row['num_heads'] != 8 or row['num_key_value_heads'] != 1
                or row['block_size'] != 128 or row['sparse_mode'] != 3
                or row['scale'] != 256 ** -.5 or row['pre_tokens'] != 2147483647
                or row['next_tokens'] != 2147483647
                or row['actual_seq_lengths'] != expected_qs
                or len(row['actual_seq_lengths_kv']) != len(expected_qs)
                or row['block_table'].shape[0] < len(expected_qs)):
            return False
    return True


class NativeFIAHostBatch:
    def __init__(self, runner, mode):
        if mode not in ('off', 'on', 'validate'):
            raise ValueError('Native FIA host batch is off, on, or validate')
        self.mode, self.enabled, self.runner = mode, mode != 'off', runner
        self.entries, self.counts, self.validations = {}, Counter(), Counter()
        self.observed = set()
        if not self.enabled:
            return
        verify_primitive()
        config = runner.vllm_config
        if (config.model_config.dtype != torch.bfloat16 or config.model_config.quantization is not None
                or config.parallel_config.tensor_parallel_size != 2 or config.parallel_config.pipeline_parallel_size != 1
                or config.parallel_config.data_parallel_size != 1 or config.lora_config is not None
                or getattr(config.model_config.hf_text_config, 'model_type', '') not in ('qwen3_5_moe', 'qwen3_5_moe_text')
                or config.speculative_config.method != 'mtp' or config.speculative_config.num_speculative_tokens != 6
                or runner._q35_gdn_input_manager.mode != mode):
            raise RuntimeError('Native FIA boxing requires the admitted Q35 BF16 TP2 dynamic4/6 lane')
        path = PRIMITIVE / 'build-r1/lib/vspec_fia_host.so'
        spec = importlib.util.spec_from_file_location('vspec_fia_host', path)
        self.library = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.library)
        if mode == 'validate':
            from vllm_ascend.attention.q35_native_host_batch_oracle import make_resolver
            self.oracle = make_resolver()

    def try_update(self, stream, context, tokens, source, attn_keys, params, handles, events,
                   workspace, metadata, draft_steps, is_draft):
        if not self.enabled:
            return False
        descriptor = getattr(context, 'batch_descriptor', None)
        if (descriptor is None or not descriptor.uniform or descriptor.num_reqs not in (1, 2, 3, 14)
                or tokens not in (descriptor.num_reqs * 5, descriptor.num_reqs * 7)
                or descriptor.num_tokens != tokens):
            return False
        requests = descriptor.num_reqs
        role = 'draft' if is_draft else 'target'
        count = 10 if role == 'target' else tokens // requests - 1
        if len(params) != count:
            return False
        if len(attn_keys) < len(params) or not len(params) == len(handles) == len(events):
            raise RuntimeError('Native FIA captured argument/handle/event cardinality changed')
        rows = derive_tasks(attn_keys, params, metadata, draft_steps, is_draft, workspace)
        if rows is None:
            return False
        if (role, tokens) not in self.observed:
            self.observed.add((role, tokens))
            geometry = [(r['query'].shape[0], len(r['actual_seq_lengths']),
                         r['actual_seq_lengths'][-1], len(r['actual_seq_lengths_kv']),
                         r['block_table'].shape[0]) for r in rows]
            print(f'[q35-native-fia-host-shapes] pid={os.getpid()} role={role} tokens={tokens} '
                  f'query_rows_qbatch_qlast_kvbatch_table_rows={geometry}', flush=True)
        if self.mode == 'validate':
            original = self.oracle(attn_keys, params, handles, events, metadata, draft_steps, is_draft, workspace)
            same_arguments(rows, original)
            self.validations[role, tokens] += 1
        identity = id(source), tokens, role
        signature = tuple(map(id, params)), tuple(map(id, handles)), tuple(map(id, events)), workspace.data_ptr()
        entry = self.entries.get(identity)
        if entry is None:
            if not eligible_rows(rows, role, tokens, requests):
                return False
            bindings = [self.library.NativeFIA(row['query'], row['key'], row['value'], row['atten_mask'],
                workspace, *row['out'], row['num_heads'], row['num_key_value_heads'], row['block_size'],
                row['sparse_mode'], row['pre_tokens'], row['next_tokens'], row['scale']) for row in rows]
            entry = (source, signature, self.library.NativeFIABatch(bindings, handles, events),
                     tuple(tuple(r['actual_seq_lengths']) for r in rows))
            self.entries[identity] = entry
            print(f'[q35-native-fia-host-batch] pid={os.getpid()} mode={self.mode} role={role} '
                  f'tokens={tokens} tasks={count} original_events_and_lengths_retained=True', flush=True)
        elif entry[0] is not source or entry[1] != signature:
            raise RuntimeError('Native FIA owned graph bindings changed after capture')
        if any(tuple(r['actual_seq_lengths']) != expected or len(r['actual_seq_lengths_kv']) != len(expected)
               or r['block_table'].shape[0] < len(expected) or r['block_table'].dtype != torch.int32
               for r, expected in zip(rows, entry[3], strict=True)):
            return False
        with torch.npu.stream(stream):
            entry[2].update(stream, [r['actual_seq_lengths'] for r in rows],
                            [r['actual_seq_lengths_kv'] for r in rows], [r['block_table'] for r in rows])
        self.counts[role, tokens] += 1
        return True

    def reset(self):
        self.counts.clear()
        self.validations.clear()

    def metrics(self):
        return dict(mode=self.mode, actual_update_calls={f'{r}:{t}': n for (r, t), n in self.counts.items()},
                    independent_original_native_argument_checks={f'{r}:{t}': n for (r, t), n in self.validations.items()},
                    cached_owned_graph_batches=len(self.entries), original_native_FIA_and_events=True,
                    no_GPU_values_read_or_new_device_synchronization=True)
